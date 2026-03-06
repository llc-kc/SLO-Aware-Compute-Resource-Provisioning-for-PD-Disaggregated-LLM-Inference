import matplotlib.pyplot as plt
import numpy as np
import logging


# user requirements
in_seq_len = 6144
out_seq_len = 512


TTFT_s = 2
TOPT_ms = 20  # ms

# user required token/s
total_throughput = 500 * 10000 / 60  # 500w TPM as an example


# benchmarked results

# ttft_overhead, mainly kv cache transfer time from P to D, unit is s
ttft_overhead = 0.1

# prefill throughput at input seq length
prefill_throughput_ult = 28309


decode_batch = np.array([
    10,
    20,
    30,
    40,
    50,
    60,
])

# TPOT, unit is ms
decode_topt = np.array([
    12.8,
    16.4,
    18.8,
    21.6,
    23.9,
    26.2,
])


# plot curve
decode_throughput = decode_batch * 1000 / decode_topt

fig, ax1 = plt.subplots(figsize=(6, 5), dpi=300)
plt.grid(True, axis="y")

ax1.plot(decode_batch, decode_topt, 'b^-')

ax1.set_xlabel('Decode batch')
ax1.set_ylabel('TOPT (ms)', color='b')


ax2 = ax1.twinx()
ax2.plot(decode_batch, decode_throughput, 'rs-')
ax2.set_ylabel('Decode throughput (token/s)', color='r')


fig, ax1 = plt.subplots(figsize=(6, 5), dpi=300)
plt.grid(True, axis="y")

ax1.plot(decode_topt, decode_batch, 'b^-')

ax1.set_xlabel('TOPT (ms)')
ax1.set_ylabel('Decode batch', color='b')


ax2 = ax1.twinx()
ax2.plot(decode_topt, decode_throughput, 'rs-')
ax2.set_ylabel('Decode throughput (token/s)', color='r')


# compute prefill throughput

ttft_ult_th = 0.95
# the miminum achievable TTFT
ttft_ult = (in_seq_len / prefill_throughput_ult + ttft_overhead) * ttft_ult_th

if TTFT_s < ttft_ult:
    logging.error(f"required TTFT {TTFT_s} s can not be achieved, reset to {ttft_ult} s")
    TTFT_s = ttft_ult


prefill_throughput = int(prefill_throughput_ult - in_seq_len/(TTFT_s - ttft_overhead))
print(f"actually achieved prefill throughput: {prefill_throughput} token/s at TTFT {TTFT_s} s and input length {in_seq_len}")


# compute decode throughput

poly_coeffs_2rd = np.polyfit(decode_topt, decode_throughput, deg=2)


def quadratic_interpolation(tpot, poly_coeffs_2rd):
    a, b, c = poly_coeffs_2rd
    tpot = np.array(tpot)
    batch = a * (tpot ** 2) + b * tpot + c
    return batch.astype("int").tolist()


decode_thrpughput = quadratic_interpolation(TOPT_ms, poly_coeffs_2rd)
print(f"actually achieved decode throughput: {decode_thrpughput} token/s at TPOT {TOPT_ms} ms and input length {in_seq_len}")

pd_ratio = in_seq_len * decode_thrpughput / (out_seq_len * prefill_throughput)

total_seq_len = in_seq_len + out_seq_len
p_num = total_throughput * in_seq_len / (total_seq_len * prefill_throughput)
d_num = total_throughput * out_seq_len / (total_seq_len * decode_thrpughput)

print(f"pd ratio: {pd_ratio:.2f}")
print(f"prefill instance num: {p_num:.2f} at total throughput {total_throughput} token/s")
print(f"decode instance num: {d_num:.2f} at total throughput {total_throughput} token/s")
