# Comparison and acceptance policy

## NCCL: two useful points

The explorer and simplified detail assessment show only **16 GiB** and **32 GiB** message sizes. Bandwidth is **GB/s**; message size is **GiB**. The displayed comparison is recorded bus bandwidth, not a mixture of bus bandwidth, algorithm bandwidth and latency.

For each size the backend selects the best independent recorded result for the matching NCCL version/configuration. Source job and placement remain attached. The percentage is:

`100 × (your GB/s / best recorded GB/s − 1)`

Positive means over the recorded reference, negative means under. A missing observed point or missing positive reference has no percentage. Missing 32 GiB evidence is not interpolated or invented. Where native element-count rounding makes actual bytes differ slightly from the nominal size label, the tooltip records the actual bytes and the comparison uses those exact bytes.

The reference can be a contextual historical cluster result when complete fabric context is unavailable. That scope is disclosed in the detail note; numerical similarity does not establish full recipe qualification. The table's short labels distinguish a comparison, a missing reference and an absent 16/32 GiB test.

The simplified view does not replace the retained raw curve or the stricter exact baseline qualification. Expand raw evidence when investigating a different size, datatype, operation, placement or correctness problem. A failed parent job keeps the result in a diagnostic state even when the individual NCCL file reports zero correctness errors.

## Exact NCCL baseline eligibility

Qualified baselines require complete source-backed results, zero correctness errors and matching scale, hardware, transport, collective, NCCL version, exact message bytes, datatype, operation and placement. Split-mask and protocol/topology evidence must be comparable. Reviewed uploaded references remain explicitly reviewed evidence rather than automatically becoming historical qualified results.

NCCL environment, test iterations/warm-up/registration, rails, NVLink domains and fabric metadata matter. Missing values remain unknown; differing known fabric/network configurations must not be merged. Reference quality is separate from arithmetic over/under.

## Training

Training references match the workload/model and relevant model size, GPU/scale, precision, batch and sequence settings, software release and parallelism dimensions. Unknown or differing context is exposed. Do not compare unrelated models as interchangeable throughput targets.

The recorded run summary can be an arithmetic mean while the benchmark comparison uses a median. Both statistics are labeled. Throughput improves upward; seconds per step improves downward. Performance ratios normalize direction before classifying the result.

The framework's default training assessment bands are within 5% below reference for good, up to 10% below for bad, and larger deficits for worst. This is an explicit application policy, not automatic acceptance under every external source document. Conditional metrics can show arithmetic ratios while remaining unscored. Unknown reference evidence is not a good result.

## Execution and final results

Execution health and performance are separate. A fast failed run is failed evidence. Scheduler/parent failure, numerical failure, nonfinite values and incomplete training prevent qualification. A source-reported scheduler or historical status is labeled as such rather than promoted to a measured correctness pass.

Final-number and Exemplar designations are explicit local actions on eligible evidence. They preserve ownership and provenance and do not certify the entire cluster or publish to an external system.
