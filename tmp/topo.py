from collections import defaultdict
import statistics
import time
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp


def map_gfc_topology():
  devices = jax.devices()
  num_devices = len(devices)

  print(f"Detected {num_devices} devices.")
  if num_devices < 2:
    print("Need at least 2 devices to benchmark topology.")
    return

  # 1. Define Mesh
  mesh = jax.sharding.Mesh(devices, ("x",))
  partition_spec = jax.sharding.PartitionSpec("x", None)

  # 2. Define Payloads
  lat_elements = 1
  bw_elements = (1024 * 1024 * 1024) // 4  # 1024 MB per device

  def measure_latency_delta(
      src: int, dst: int, n1: int = 500, n2: int = 2500, num_trials: int = 5
  ):
    global_shape = (num_devices, lat_elements)
    sharding = jax.sharding.NamedSharding(mesh, partition_spec)

    key = jax.random.PRNGKey(0)
    data = jax.jit(
        lambda k: jax.random.normal(k, global_shape, dtype=jnp.float32),
        out_shardings=sharding,
    )(key)

    @jax.jit
    def ping_pong_step(x, iters):
      def ppermute_kernel(val, iters_local):
        def body_fn(i, v):
          # Sequential Ping-Pong
          v_ping = jax.lax.ppermute(v, axis_name="x", perm=[(src, dst)])
          v_pong = jax.lax.ppermute(v_ping, axis_name="x", perm=[(dst, src)])
          return v_pong + 1e-5

        return jax.lax.fori_loop(0, iters_local, body_fn, val)

      return shard_map(
          ppermute_kernel,
          mesh=mesh,
          # Pass 'iters' as a fully replicated scalar
          in_specs=(partition_spec, jax.sharding.PartitionSpec()),
          out_specs=partition_spec,
          check_rep=False,
      )(x, iters)

    # Warmup
    ping_pong_step(data, jnp.int32(1)).block_until_ready()

    def get_median_time(iters):
      trial_times_s = []
      for _ in range(num_trials):
        start_time = time.perf_counter()
        res = ping_pong_step(data, jnp.int32(iters))
        res.block_until_ready()
        trial_times_s.append(time.perf_counter() - start_time)
      return statistics.median(trial_times_s)

    # Two-point delta
    t1_s = get_median_time(n1)
    t2_s = get_median_time(n2)

    delta_t_s = max(t2_s - t1_s, 1e-9)
    delta_iters = n2 - n1

    # 1 iteration = 2 one-way hops (Ping + Pong)
    total_delta_hops = delta_iters * 2
    avg_time_us = (delta_t_s / total_delta_hops) * 1e6
    return avg_time_us

  def measure_uni_bandwidth_delta(
      src: int, dst: int, n1: int = 5, n2: int = 25, num_trials: int = 5
  ):
    global_shape = (num_devices, bw_elements)
    sharding = jax.sharding.NamedSharding(mesh, partition_spec)

    key = jax.random.PRNGKey(1)
    data = jax.jit(
        lambda k: jax.random.normal(k, global_shape, dtype=jnp.float32),
        out_shardings=sharding,
    )(key)

    @jax.jit
    def uni_transfer_step(x, iters):
      def ppermute_kernel(val, iters_local):
        def body_fn(i, v):
          # Unidirectional
          return jax.lax.ppermute(v, axis_name="x", perm=[(src, dst)])

        return jax.lax.fori_loop(0, iters_local, body_fn, val)

      return shard_map(
          ppermute_kernel,
          mesh=mesh,
          in_specs=(partition_spec, jax.sharding.PartitionSpec()),
          out_specs=partition_spec,
          check_rep=False,
      )(x, iters)

    # Warmup
    uni_transfer_step(data, jnp.int32(1)).block_until_ready()

    def get_median_time(iters):
      trial_times_s = []
      for _ in range(num_trials):
        start_time = time.perf_counter()
        res = uni_transfer_step(data, jnp.int32(iters))
        res.block_until_ready()
        trial_times_s.append(time.perf_counter() - start_time)
      return statistics.median(trial_times_s)

    # Two-point delta
    t1_s = get_median_time(n1)
    t2_s = get_median_time(n2)

    delta_t_s = max(t2_s - t1_s, 1e-9)
    delta_iters = n2 - n1

    # Unidirectional payload
    bytes_per_iter = bw_elements * 4
    avg_time_s_per_iter = delta_t_s / delta_iters
    bandwidth_gb_s = (bytes_per_iter / avg_time_s_per_iter) / 1e9

    return bandwidth_gb_s

  # 3. Gather Measurements
  print(f"Measuring for {num_devices * (num_devices - 1)} pairs...")
  print("-" * 85)
  results = defaultdict(dict)

  for src in range(num_devices):
    for dst in range(num_devices):
      if src == dst:
        continue

      lat_us = measure_latency_delta(src, dst)
      bw_gbs = measure_uni_bandwidth_delta(src, dst)

      results[src][dst] = {"lat": lat_us, "bw": bw_gbs}
      print(
          f"  Dev {src} -> Dev {dst:<3} | Uni Latency: {lat_us:>6.2f} us | Uni"
          f" Bandwidth: {bw_gbs:>7.2f} GB/s"
      )

  # 4. Topology Discovery via Chip-Aware Clustering & Sub-Die Affinity
  print("\n" + "=" * 85)
  print(" MEASUREMENTS")
  print("=" * 85)

  # Step A: Identify physical chips via D2D (lowest latency partner)
  d2d_partner = {}
  for src in range(num_devices):
    dsts_sorted = sorted(
        results[src].keys(), key=lambda d: results[src][d]["lat"]
    )
    d2d_partner[src] = dsts_sorted[0]

  chips = set()
  for i in range(num_devices):
    chips.add(tuple(sorted((i, d2d_partner[i]))))

  for src in range(num_devices):
    my_chip = next(c for c in chips if src in c)
    partner = d2d_partner[src]

    # Step B: Calculate average latency to each remote chip
    chip_lats = {}
    for c in chips:
      if c == my_chip:
        continue
      # Average the latency of the two cores on the destination chip
      avg_lat = (results[src][c[0]]["lat"] + results[src][c[1]]["lat"]) / 2.0
      chip_lats[c] = avg_lat

    # Step C: Sort chips and find direct/indirect ICI neighbors
    sorted_remote_chips = sorted(chip_lats.keys(), key=lambda c: chip_lats[c])
    sorted_lats = [chip_lats[c] for c in sorted_remote_chips]

    direct_ici_chips = []
    multi_hop_chips = []

    if len(sorted_lats) > 1:
      max_gap = 0
      split_idx = len(sorted_lats)
      for i in range(1, len(sorted_lats)):
        gap = sorted_lats[i] - sorted_lats[i - 1]
        if gap > max_gap:
          max_gap = gap
          split_idx = i

      if max_gap > 0.3:
        direct_ici_chips = sorted_remote_chips[:split_idx]
        multi_hop_chips = sorted_remote_chips[split_idx:]
      else:
        direct_ici_chips = sorted_remote_chips
    elif len(sorted_lats) == 1:
      direct_ici_chips = sorted_remote_chips

    # Flatten chips back into devices
    direct_devs = [d for c in direct_ici_chips for d in c]
    multi_devs = [d for c in multi_hop_chips for d in c]

    # Step D: Determine ICI-connected cores
    if direct_devs:
      avg_lat_src_to_ici = sum(
          results[src][d]["lat"] for d in direct_devs
      ) / len(direct_devs)
      avg_lat_partner_to_ici = sum(
          results[partner][d]["lat"] for d in direct_devs
      ) / len(direct_devs)

      if avg_lat_src_to_ici < avg_lat_partner_to_ici:
        core_type = "[ICI-Adjacent Core]"
      elif avg_lat_src_to_ici > avg_lat_partner_to_ici:
        core_type = "[Non-Adjacent Core]"
      else:
        core_type = "[Unknown Core].    "
    else:
      core_type = "[Unknown Core]     "

    # Sort devices within categories by latency
    direct_devs.sort(key=lambda d: results[src][d]["lat"])
    multi_devs.sort(key=lambda d: results[src][d]["lat"])

    print(f"\nDevice {src} {core_type}:")
    print(
        f"  [D2D / Same Chip]      Dev {partner:2d} (Uni Lat:"
        f" {results[src][partner]['lat']:6.2f} us, Uni BW:"
        f" {results[src][partner]['bw']:8.2f} GB/s)"
    )

    for d in direct_devs:
      print(
          f"  [Direct ICI Neighbor]  Dev {d:2d} (Uni Lat:"
          f" {results[src][d]['lat']:6.2f} us, Uni BW:"
          f" {results[src][d]['bw']:8.2f} GB/s)"
      )

    for d in multi_devs:
      print(
          f"  [Multi-Hop]            Dev {d:2d} (Uni Lat:"
          f" {results[src][d]['lat']:6.2f} us, Uni BW:"
          f" {results[src][d]['bw']:8.2f} GB/s)"
      )


if __name__ == "__main__":
  map_gfc_topology()