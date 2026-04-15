# HIP Graph PM4 Dispatch Experiment Guide

## Background: Why PM4?

AMD GPUs have two compute queue formats understood by the Command Processor (CP/MEC firmware):

| Queue Type | KFD ioctl | Description |
|---|---|---|
| `KFD_IOC_QUEUE_TYPE_COMPUTE` (0x0) | PM4 native | CP directly executes PM4 commands from ring buffer |
| `KFD_IOC_QUEUE_TYPE_COMPUTE_AQL` (0x2) | AQL (HSA) | MEC firmware reads 64-byte AQL packets, translates each into PM4 microinstructions, then dispatches |

Today, all HIP kernel dispatches (both eager and graph) go through AQL. The MEC firmware
translation from AQL → PM4 adds ~0.5–1μs per dispatch. For workloads with thousands of tiny
kernels, this overhead is significant.

The hypothesis: **if we submit PM4 packets directly (bypassing AQL), we eliminate the MEC
translation overhead and get faster kernel dispatch**.

tinygrad already does this on their custom stack. This experiment validates the approach
within the ROCm ecosystem.

---

## CRITICAL: CDNA4 / MI355 / gfx950 — AQL is STILL Firmware-Emulated

### Finding: gfx950 is GFX 9.5.0, NOT GFX12

CDNA4 (MI350/MI355) in this codebase is identified as **`gfx950` with ISA version 9.5.0**,
NOT gfx12. GFX12 is RDNA4 (consumer GPUs like gfx1200/gfx1201). This is confirmed by:

```
// projects/rocr-runtime/runtime/hsa-runtime/core/runtime/isa.cpp
ISAREG_ENTRY_GEN("gfx950", 9, 5, 0, any, any, 64, "gfx9-4-generic")
```

### AQL Processing on ALL current AMD GPUs: MEC Firmware Decode

Based on tinygrad's reverse-engineering of MEC firmware and the code in this repo,
**AQL packets are NEVER "natively" decoded by hardware**. On ALL current AMD GPUs
(GFX7 through GFX12, including gfx950/CDNA4), the process is:

1. User writes AQL packet (64 bytes) to the AQL ring buffer
2. User rings the doorbell (MMIO write)
3. **MEC firmware** (a microcontroller running on the GPU) reads the AQL packet
4. **MEC firmware parses each field** and writes to COMPUTE_* registers via register stores
5. MEC writes `COMPUTE_DISPATCH_INITIATOR` which triggers the actual dispatch

From tinygrad's disassembly of MEC firmware (polaris10, gc_11_0_1):
```
DISPATCH_DIRECT:
    stw r6, reg[r0, #0x2e01]  # COMPUTE_DIM_X
    stw r4, reg[r0, #0x2e02]  # COMPUTE_DIM_Y
    stw r5, reg[r0, #0x2e03]  # COMPUTE_DIM_Z
    stw r3, reg[r0, #0x2e00]  # COMPUTE_DISPATCH_INITIATOR (triggers dispatch)
```

The `AQL_CONTROL` register bit (`CP_HQD_AQL_CONTROL == 0x1`) tells the MEC "this queue
uses AQL format" so it knows to parse 64-byte AQL packets instead of variable-length PM4.

### The `AqlEmulationPm4_` Flag — What It Really Means

The codebase has a `PM4_EMULATION` flag, but it means the OPPOSITE of what you'd expect:

```c
// libhsakmt/include/hsakmt/hsakmttypes.h
unsigned int AqlEmulationPm4_ : 1; // Indicates device uses AQL emulation via PM4 packets
```

This flag is `1` when AQL is being **emulated on top of PM4** (the Windows/DXG path
where the runtime manually translates AQL→PM4 in software, then submits PM4 to the queue).
On Linux with hardware HWS (Hardware Scheduler), this flag is typically `0` because
AQL queues are handled by MEC firmware directly.

The key code:
```c
// libhsakmt/src/dxg/topology.cpp
props.Capability2.ui32.AqlEmulationPm4_ =
    (device->IsAqlSupported() && device->DeviceInfo().hwsInfo.hwsMask.computeHwsEnabled) ? 0 : 1;
```

- `AqlEmulationPm4_ = 0` → "hardware supports AQL queues (via MEC firmware)"
- `AqlEmulationPm4_ = 1` → "no AQL HW support, runtime must emulate AQL using PM4"

**Neither case has true "hardware native" AQL decode.** The `0` case means MEC firmware
does the decode; the `1` case means the host-side software does it.

### Implications for MI355 / gfx950

On your MI355 (gfx950), running Linux with KFD:
- KFD creates `KFD_IOC_QUEUE_TYPE_COMPUTE_AQL` queues
- `AqlEmulationPm4_` is likely `0` (MEC firmware handles AQL)
- **MEC firmware still does the AQL→register-write translation for every dispatch**
- The PM4 experiment is therefore **still relevant** for MI355

### Verification: Run This on Your MI355

```bash
# Check the capability2 flag
cat /sys/class/kfd/kfd/topology/nodes/*/properties | grep capability2

# Check MEC firmware version
cat /sys/class/drm/card*/device/fw_version/mec_fw_version

# Dump queue state (if umr is available)
sudo umr -cpc | grep AQL_CONTROL
# AQL_CONTROL == 0x1 means the queue is in AQL mode (firmware decode)
# AQL_CONTROL == 0x0 means the queue is in PM4 mode
```

---

## Architecture: AQL vs PM4 Dispatch Paths

### Current AQL path (eager or graph)

```
hipLaunchKernel / hipGraphLaunch
  → CLR: build hsa_kernel_dispatch_packet_t (64 bytes)
    → memcpy to AQL ring buffer (gpu_queue_->base_address)
      → doorbell MMIO write
        → MEC firmware reads AQL ring
          → MEC translates AQL fields → PM4 microinstructions  [~0.5-1μs overhead]
            → Shader Engine dispatch
```

### Proposed PM4 path

```
PM4 Dispatch Benchmark
  → Build PM4 command stream:
      SET_SH_REG(COMPUTE_PGM_LO/HI)      - shader address
      SET_SH_REG(COMPUTE_PGM_RSRC1/2)     - resources
      SET_SH_REG(COMPUTE_NUM_THREAD_X/Y/Z) - workgroup size
      SET_SH_REG(COMPUTE_USER_DATA_0+)    - kernarg pointer, etc.
      DISPATCH_DIRECT(grid_x, grid_y, grid_z, initiator)
    → Submit via AqlQueue::ExecutePM4 (vendor AQL slot + IB jump)
      → or via raw PM4 queue (HSA_QUEUE_COMPUTE)
        → doorbell MMIO write
          → CP executes PM4 directly  [NO translation overhead]
            → Shader Engine dispatch
```

---

## Repository Structure (What You Need)

All code lives in this monorepo. The relevant projects:

```
projects/
├── clr/                          # ROCm CLR (HIP runtime + ROCclr device layer)
│   ├── hipamd/src/               # HIP API implementation
│   │   ├── hip_graph.cpp         # Graph capture/instantiate/launch entry points
│   │   ├── hip_graph_internal.hpp  # Graph/GraphExec/GraphNode class definitions
│   │   ├── hip_graph_internal.cpp  # GraphExec::Run, CaptureAQLPackets, EnqueueSegment
│   │   ├── hip_graph_capture.hpp   # Capture hook declarations
│   │   ├── hip_module.cpp          # hipLaunchKernel (eager path)
│   │   └── hip_platform.cpp        # ihipLaunchKernel
│   └── rocclr/device/rocm/      # ROCclr GPU backend
│       ├── rocvirtual.cpp        # VirtualGPU: dispatchAqlPacket, dispatchAqlPacketBatchFlat,
│       │                         #   submitKernelInternal, dispatchCounterAqlPacket
│       ├── rocvirtual.hpp        # VirtualGPU declarations
│       ├── rocdevice.cpp         # Device init, PM4 emulation flag, CreateBarrierPacket
│       └── rocdevice.hpp         # Device class, IsPm4Emulation()
│
├── rocr-runtime/                 # ROCr (HSA runtime + KFD thunk)
│   ├── runtime/hsa-runtime/
│   │   ├── core/runtime/
│   │   │   ├── amd_aql_queue.cpp   # AqlQueue::ExecutePM4 - EXISTING PM4 submission path
│   │   │   ├── amd_gpu_agent.cpp   # Queue creation, PM4 cache flush commands
│   │   │   └── hsa.cpp             # hsa_queue_create entry point
│   │   ├── core/inc/
│   │   │   └── amd_gpu_pm4.h       # PM4 packet header macros (NOP, INDIRECT_BUFFER,
│   │   │                           #   RELEASE_MEM, ACQUIRE_MEM, etc.)
│   │   ├── inc/
│   │   │   ├── amd_hsa_kernel_code.h  # amd_kernel_code_t (kernel descriptor with
│   │   │   │                           #   compute_pgm_rsrc1/2, kernel_code_properties)
│   │   │   ├── amd_hsa_queue.h        # amd_queue_t struct (read/write dispatch IDs)
│   │   │   └── hsa_ext_amd.h          # HSA AMD extensions
│   │   └── loader/
│   │       └── AMDHSAKernelDescriptor.h  # V3 kernel descriptor (amdhsa::kernel_descriptor_t)
│   │
│   └── libhsakmt/                # KFD thunk layer
│       ├── include/hsakmt/
│       │   ├── hsakmttypes.h       # HSA_QUEUE_COMPUTE=1 (PM4), HSA_QUEUE_COMPUTE_AQL=21
│       │   └── linux/kfd_ioctl.h   # KFD_IOC_QUEUE_TYPE_COMPUTE=0x0, COMPUTE_AQL=0x2
│       ├── include/impl/
│       │   └── pm4_cmds.h          # PM4 command structs: PM4_MEC_SET_SH_REG,
│       │                           #   PM4_MEC_DISPATCH_DIRECT, mmCOMPUTE_* register offsets
│       ├── src/queues.c            # hsaKmtCreateQueueV2 (KFD ioctl wrapper)
│       └── tests/kfdtest/          # KFD test suite with PM4 queue examples
│           ├── src/PM4Packet.cpp   # PM4 packet construction (WRITE_DATA, RELEASE_MEM, etc.)
│           ├── src/PM4Queue.cpp    # Raw PM4 queue: wptr/doorbell submission
│           ├── include/pm4_pkt_struct_common.h  # PM4_DISPATCH_DIRECT struct
│           └── include/kfd_pm4_opcodes.h        # IT_DISPATCH_DIRECT=0x15
│
├── hip-tests/catch/              # HIP test suite
│   ├── perftests/graph/
│   │   ├── hipPerfGraphLaunch.cc   # Graph launch benchmark
│   │   ├── parallelGraph.cc        # Parallel graph perf tests
│   │   └── hipGraphTopology.cc     # Graph topology perf tests
│   └── unit/graph/                 # Graph unit tests (capture, launch, etc.)
│
├── aqlprofile/                   # AQL profiling library
│   ├── linux/packets/
│   │   ├── soc15d.h              # PACKET3_DISPATCH_DIRECT=0x15, PACKET3_SET_SH_REG=0x76
│   │   └── nvd.h                 # Same constants (alternate naming)
│   ├── linux/registers/gc/       # Per-ASIC register offset/mask headers
│   │   ├── gc_9_2_1_offset.h     # GFX9: mmCOMPUTE_DISPATCH_INITIATOR=0x0e00, etc.
│   │   ├── gc_9_4_2_offset.h     # MI200 variant
│   │   ├── gc_9_4_2_sh_mask.h    # Bitmask definitions
│   │   ├── gc_10_3_0_offset.h    # GFX10
│   │   ├── gc_11_0_0_offset.h    # GFX11
│   │   └── gc_12_0_0_offset.h    # GFX12
│   └── src/pm4/
│       ├── gfx9_cmd_builder.h    # PM4 command builders using SET_SH_REG (profiling)
│       ├── gfx10_cmd_builder.h
│       ├── gfx11_cmd_builder.h
│       └── gfx12_cmd_builder.h
│
└── rocr-runtime/libhsakmt/src/dxg/wddm/
    └── cmd_util.cpp              # WDDM dispatch: complete AQL→PM4 translation reference
                                  #   BuildDispatch() shows full SET_SH_REG + DISPATCH_DIRECT
                                  #   sequence with all registers
```

---

## Key Data Structures

### AQL Kernel Dispatch Packet (64 bytes) - what we use today

```c
// From HSA spec / hsa.h
typedef struct hsa_kernel_dispatch_packet_s {
    uint16_t header;              // packet type + fence scopes
    uint16_t setup;               // dimensions
    uint16_t workgroup_size_x;
    uint16_t workgroup_size_y;
    uint16_t workgroup_size_z;
    uint16_t reserved0;
    uint32_t grid_size_x;
    uint32_t grid_size_y;
    uint32_t grid_size_z;
    uint32_t private_segment_size;
    uint32_t group_segment_size;
    uint64_t kernel_object;       // GPU address of kernel code
    uint64_t kernarg_address;     // GPU address of kernel arguments
    uint64_t reserved2;
    uint64_t completion_signal;
} hsa_kernel_dispatch_packet_t;   // 64 bytes total
```

### PM4 Dispatch Sequence (what we want to build)

A complete PM4 kernel dispatch consists of multiple PM4 commands chained together:

```c
// 1. SET_SH_REG: workgroup dimensions (3 dwords payload)
//    Register: COMPUTE_NUM_THREAD_X (offset 0x0e07 on GFX9)
PM4_TYPE3_HEADER(IT_SET_SH_REG, 5)   // header: 1 dword
reg_offset = COMPUTE_NUM_THREAD_X - 0x2c00  // 1 dword
compute_num_thread_x = workgroup_size_x     // 1 dword
compute_num_thread_y = workgroup_size_y     // 1 dword  
compute_num_thread_z = workgroup_size_z     // 1 dword
                                            // Total: 5 dwords = 20 bytes

// 2. SET_SH_REG: program address (2 dwords payload)
//    Register: COMPUTE_PGM_LO (offset 0x0e0c on GFX9)
PM4_TYPE3_HEADER(IT_SET_SH_REG, 4)
reg_offset = COMPUTE_PGM_LO - 0x2c00
compute_pgm_lo = kernel_code_addr >> 8      // 48-bit address, low 32
compute_pgm_hi = kernel_code_addr >> 40     // high 8 bits
                                            // Total: 4 dwords = 16 bytes

// 3. SET_SH_REG: program resources (2 dwords payload)
//    Register: COMPUTE_PGM_RSRC1 (offset 0x0e12 on GFX9)
PM4_TYPE3_HEADER(IT_SET_SH_REG, 4)
reg_offset = COMPUTE_PGM_RSRC1 - 0x2c00
compute_pgm_rsrc1 = <from kernel descriptor>  // vgprs, sgprs, float_mode
compute_pgm_rsrc2 = <from kernel descriptor>  // LDS, scratch, etc.
                                              // Total: 4 dwords = 16 bytes

// 4. SET_SH_REG: resource limits + thread management (6 dwords payload)
//    Register: COMPUTE_RESOURCE_LIMITS (offset 0x0e15 on GFX9)
PM4_TYPE3_HEADER(IT_SET_SH_REG, 8)
reg_offset = COMPUTE_RESOURCE_LIMITS - 0x2c00
compute_resource_limits = 0x3ff
compute_static_thread_mgmt_se0 = 0xFFFFFFFF
compute_static_thread_mgmt_se1 = 0xFFFFFFFF
compute_static_thread_mgmt_se2 = 0xFFFFFFFF
compute_static_thread_mgmt_se3 = 0xFFFFFFFF
compute_tmpring_size = <scratch config>
                                              // Total: 8 dwords = 32 bytes

// 5. SET_SH_REG: user data (kernarg pointer, etc.)
//    Register: COMPUTE_USER_DATA_0 (offset 0x0e40 on GFX9)
PM4_TYPE3_HEADER(IT_SET_SH_REG, 2 + N_sgprs)
reg_offset = COMPUTE_USER_DATA_0 - 0x2c00
user_data[0] = kernarg_addr_lo   // if ENABLE_SGPR_KERNARG_SEGMENT_PTR
user_data[1] = kernarg_addr_hi
// ... other implicit SGPRs based on kernel_code_properties
                                 // Total: (2+N) dwords

// 6. DISPATCH_DIRECT (4 dwords payload)
PM4_TYPE3_HEADER(IT_DISPATCH_DIRECT, 5)
dim_x = grid_size_x
dim_y = grid_size_y
dim_z = grid_size_z
dispatch_initiator = COMPUTE_SHADER_EN | FORCE_START_AT_000 | USE_THREAD_DIMENSIONS
                   | (wave32 ? CS_W32_EN : 0)
                                 // Total: 5 dwords = 20 bytes

// TOTAL: ~30-40 dwords (120-160 bytes) vs AQL's 64 bytes
// But: NO MEC translation needed
```

### PM4 Header Format

```c
// From pm4_cmds.h
#define PM4_TYPE3_HDR(_opc_, _count_) \
    (uint32_t)((3)                 << 30 | \   // type = 3
               ((_count_) - 2)     << 16 | \   // dword count minus 2
               (_opc_)             << 8)  | \   // opcode
               (PM4_COMPUTE_SHADER << 1)        // shader type = compute

// Key opcodes:
#define IT_DISPATCH_DIRECT  0x15
#define IT_SET_SH_REG       0x76
#define IT_RELEASE_MEM      0x49
#define IT_ACQUIRE_MEM      0x58
```

### Kernel Descriptor (where to get RSRC1/RSRC2)

```c
// From amd_hsa_kernel_code.h - the V1 "kernel code object" format
typedef struct amd_kernel_code_s {
    // ...
    int64_t  kernel_code_entry_byte_offset;   // offset from descriptor to actual code
    // ...
    uint32_t compute_pgm_rsrc1;               // VGPR/SGPR counts, float mode, etc.
    uint32_t compute_pgm_rsrc2;               // LDS, scratch enable, etc.
    uint32_t kernel_code_properties;          // which implicit SGPRs to load
    uint32_t workitem_private_segment_byte_size;
    uint32_t workgroup_group_segment_byte_size;
    // ...
} amd_kernel_code_t;

// For V3 kernel descriptors (GFX9+), see AMDHSAKernelDescriptor.h:
//   amdhsa::kernel_descriptor_t with similar fields
```

---

## Experiment Plan

### Phase 0: Baseline Measurements (No Code Changes)

**Goal**: Measure the actual MEC translation overhead on your GPU.

**Test 1: Empty Kernel Dispatch Latency**

```cpp
// benchmark_dispatch_latency.cpp
// Measure per-kernel dispatch overhead for eager, graph, and PM4-via-vendor-AQL paths

#include <hip/hip_runtime.h>
#include <chrono>

__global__ void empty_kernel() {}

int main() {
    hipStream_t stream;
    hipStreamCreate(&stream);

    // Warmup
    for (int i = 0; i < 100; i++) {
        empty_kernel<<<1, 1, 0, stream>>>();
    }
    hipStreamSynchronize(stream);

    constexpr int N = 10000;

    // === Test 1: Eager dispatch latency ===
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        empty_kernel<<<1, 1, 0, stream>>>();
    }
    hipStreamSynchronize(stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    double eager_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;

    // === Test 2: Graph dispatch latency ===
    hipGraph_t graph;
    hipGraphExec_t graphExec;

    hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    empty_kernel<<<1, 1, 0, stream>>>();
    hipStreamEndCapture(stream, &graph);
    hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    // Warmup graph
    for (int i = 0; i < 100; i++) {
        hipGraphLaunch(graphExec, stream);
    }
    hipStreamSynchronize(stream);

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        hipGraphLaunch(graphExec, stream);
    }
    hipStreamSynchronize(stream);
    auto t3 = std::chrono::high_resolution_clock::now();
    double graph_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N;

    // === Test 3: Graph with many kernels (amortize host overhead) ===
    hipGraph_t graph_multi;
    hipGraphExec_t graphExec_multi;
    constexpr int K = 100;  // kernels per graph

    hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    for (int i = 0; i < K; i++) {
        empty_kernel<<<1, 1, 0, stream>>>();
    }
    hipStreamEndCapture(stream, &graph_multi);
    hipGraphInstantiate(&graphExec_multi, graph_multi, nullptr, nullptr, 0);

    for (int i = 0; i < 100; i++) {
        hipGraphLaunch(graphExec_multi, stream);
    }
    hipStreamSynchronize(stream);

    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N / K; i++) {
        hipGraphLaunch(graphExec_multi, stream);
    }
    hipStreamSynchronize(stream);
    auto t5 = std::chrono::high_resolution_clock::now();
    double graph_multi_us = std::chrono::duration<double, std::micro>(t5 - t4).count() / (N / K) / K;

    printf("=== Dispatch Latency (per kernel, %d iterations) ===\n", N);
    printf("Eager single kernel:     %.3f us\n", eager_us);
    printf("Graph single kernel:     %.3f us\n", graph_us);
    printf("Graph %d-kernel batch:   %.3f us/kernel\n", K, graph_multi_us);

    hipGraphExecDestroy(graphExec);
    hipGraphDestroy(graph);
    hipGraphExecDestroy(graphExec_multi);
    hipGraphDestroy(graph_multi);
    hipStreamDestroy(stream);
    return 0;
}
```

Build and run:
```bash
/opt/rocm/bin/hipcc benchmark_dispatch_latency.cpp -o benchmark_dispatch_latency
./benchmark_dispatch_latency
```

**Test 2: Use rocprof for hardware-level timing**
```bash
rocprof --hip-trace --hsa-trace ./benchmark_dispatch_latency
# Examine the trace JSON for per-dispatch GPU-side timestamps
```

### Phase 1: PM4 Dispatch via ExecutePM4 (Minimal Changes)

**Goal**: Build a PM4 DISPATCH_DIRECT command and submit it through the existing
`AqlQueue::ExecutePM4` path. This still goes through an AQL queue slot (vendor packet)
but the kernel dispatch itself is PM4, not translated AQL.

This requires a standalone C++ benchmark that:
1. Creates a HIP kernel (to get the kernel code object address, RSRC1/RSRC2)
2. Allocates kernarg memory
3. Builds a PM4 command stream (SET_SH_REG + DISPATCH_DIRECT)
4. Submits via HSA vendor-specific AQL packet (similar to ExecutePM4)
5. Measures latency

**Key challenge**: Accessing `AqlQueue::ExecutePM4` from HIP. Options:
- (a) Write the benchmark at the ROCr/HSA level (use `hsa_queue_*` APIs directly)
- (b) Add a new `dispatchPm4Batch` method in CLR's VirtualGPU

Recommended: start with option (a) — a pure HSA-level benchmark.

### Phase 2: Raw PM4 Queue (Bypass AQL Entirely)

**Goal**: Create a `KFD_IOC_QUEUE_TYPE_COMPUTE` (PM4) queue through the thunk layer and
submit PM4 commands directly.

This bypasses AQL completely but requires:
- Direct KFD thunk calls (`hsaKmtCreateQueueV2` with `HSA_QUEUE_COMPUTE`)
- Manual ring buffer management (wptr/rptr instead of AQL dispatch IDs)
- The kfdtest PM4Queue code is the reference implementation

---

## Key Reference Files for PM4 Dispatch

### PM4 Command Definitions
```
projects/rocr-runtime/libhsakmt/include/impl/pm4_cmds.h
  - PM4_MEC_SET_SH_REG, PM4_MEC_DISPATCH_DIRECT structs
  - mmCOMPUTE_* register offsets (legacy 0x2E** indices)
  - IT_* opcode constants

projects/rocr-runtime/runtime/hsa-runtime/core/inc/amd_gpu_pm4.h
  - PM4_HDR macro (builds type-3 PM4 headers)
  - PM4_INDIRECT_BUFFER_* macros (for IB submission)
  - PM4_RELEASE_MEM_* macros (for completion signals)
  - PM4_ACQUIRE_MEM_* macros (for cache coherence)
```

### Register Definitions (Per-ASIC)
```
projects/aqlprofile/linux/registers/gc/
  gc_9_2_1_offset.h   → MI25/Vega (GFX9)
  gc_9_4_2_offset.h   → MI200 (GFX9)
  gc_10_3_0_offset.h  → RDNA2 (GFX10)
  gc_11_0_0_offset.h  → RDNA3 (GFX11)
  gc_12_0_0_offset.h  → RDNA4 (GFX12)

Each *_offset.h has:
  mmCOMPUTE_DISPATCH_INITIATOR  (0x0e00 on GFX9)
  mmCOMPUTE_DIM_X/Y/Z           (0x0e01-0x0e03)
  mmCOMPUTE_NUM_THREAD_X/Y/Z    (0x0e07-0x0e09)
  mmCOMPUTE_PGM_LO/HI           (0x0e0c-0x0e0d)
  mmCOMPUTE_PGM_RSRC1/2         (0x0e12-0x0e13)
  mmCOMPUTE_RESOURCE_LIMITS      (0x0e15)
  mmCOMPUTE_USER_DATA_0..15      (0x0e40-0x0e4f)

Each *_sh_mask.h has:
  COMPUTE_DISPATCH_INITIATOR__COMPUTE_SHADER_EN__SHIFT = 0x0
  COMPUTE_DISPATCH_INITIATOR__FORCE_START_AT_000__SHIFT = 0x2
  COMPUTE_DISPATCH_INITIATOR__USE_THREAD_DIMENSIONS__SHIFT = 0x5
  COMPUTE_DISPATCH_INITIATOR__CS_W32_EN__SHIFT = 0xf  (GFX10+)
```

### Existing PM4 Submission
```
projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp
  AqlQueue::ExecutePM4()
  - Lines 1547-1675
  - GFX8: raw PM4 stuffed into AQL slot (NOP + IB_JUMP + RELEASE_MEM)
  - GFX9+: vendor-specific AQL packet with AMD_AQL_FORMAT_PM4_IB

projects/rocr-runtime/libhsakmt/src/dxg/wddm/cmd_util.cpp
  CmdUtil::BuildDispatch()
  - Lines 188-297
  - Complete AQL→PM4 translation: SET_SH_REG for all compute registers,
    then DISPATCH_DIRECT. This is the REFERENCE IMPLEMENTATION.
```

### KFD PM4 Queue Reference
```
projects/rocr-runtime/libhsakmt/tests/kfdtest/
  src/PM4Queue.cpp        - Raw PM4 queue submission (wptr + doorbell)
  src/PM4Packet.cpp       - PM4 packet builders
  src/PM4Packet.hpp       - PM4 packet class hierarchy
  include/pm4_pkt_struct_common.h  - PM4 struct layouts
  include/kfd_pm4_opcodes.h        - PM4 opcode enum
```

### Kernel Descriptor
```
projects/rocr-runtime/runtime/hsa-runtime/inc/amd_hsa_kernel_code.h
  - amd_kernel_code_t: V1 descriptor with compute_pgm_rsrc1/2, kernel_code_properties

projects/rocr-runtime/runtime/hsa-runtime/loader/AMDHSAKernelDescriptor.h
  - amdhsa::kernel_descriptor_t: V3 descriptor (GFX9+ kernels compiled with -mcode-object-version=3+)
```

---

## Long Prompt for GPU Machine

Copy-paste this to continue the experiment on a machine with an AMD GPU:

---

```
I am researching whether replacing AQL kernel dispatch packets with raw PM4 commands
can reduce dispatch latency in HIP Graph (and eager) paths. The MEC firmware currently
translates every 64-byte AQL packet into PM4 microinstructions, costing ~0.5-1μs per
dispatch.

## What I Know

### The AQL dispatch path (current):
- hipLaunchKernel / hipGraphLaunch
  → CLR builds hsa_kernel_dispatch_packet_t (64 bytes)
  → memcpy to AQL ring (gpu_queue_->base_address)
  → doorbell MMIO write
  → MEC firmware reads AQL, translates to PM4, dispatches to Shader Engine

### The PM4 dispatch path (proposed):
- Build PM4 command stream: SET_SH_REG (shader addr, RSRC1/2, workgroup dims,
  kernarg, resource limits) + DISPATCH_DIRECT (grid dims, dispatch_initiator)
- Submit to GPU. Two sub-options:
  (a) Via AqlQueue::ExecutePM4: packs PM4 into an Indirect Buffer, submits as
      vendor-specific AQL slot. Still goes through AQL queue but MEC just does
      an IB jump, no field-by-field translation.
  (b) Via raw PM4 queue: create KFD_IOC_QUEUE_TYPE_COMPUTE (not AQL), write PM4
      directly to ring, doorbell. Completely bypasses AQL.

### Key code locations in this repo:
- AQL graph dispatch: projects/clr/hipamd/src/hip_graph_internal.cpp
  (GraphExec::Run → EnqueueSegmentedGraph → EnqueueSegment → dispatchAqlPacketBatchFlat)
- AQL single dispatch: projects/clr/rocclr/device/rocm/rocvirtual.cpp
  (submitKernelInternal → dispatchGenericAqlPacket: ring write + doorbell)
- ExecutePM4: projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp
  lines 1547-1675 (GFX9+: vendor AQL + IB jump to PM4 commands)
- PM4 command defs: projects/rocr-runtime/libhsakmt/include/impl/pm4_cmds.h
  (PM4_MEC_SET_SH_REG, PM4_MEC_DISPATCH_DIRECT, mmCOMPUTE_* register offsets)
- WDDM dispatch builder (REFERENCE for AQL→PM4 translation):
  projects/rocr-runtime/libhsakmt/src/dxg/wddm/cmd_util.cpp lines 188-297
  (BuildDispatch: SET_SH_REG for all registers + DISPATCH_DIRECT)
- Register offsets per ASIC: projects/aqlprofile/linux/registers/gc/gc_*_offset.h
- KFD PM4 queue test: projects/rocr-runtime/libhsakmt/tests/kfdtest/src/PM4Queue.cpp
- PM4 packet structs: projects/rocr-runtime/libhsakmt/tests/kfdtest/include/pm4_pkt_struct_common.h
- PM4 opcodes: projects/rocr-runtime/libhsakmt/include/impl/pm4_cmds.h
  (IT_DISPATCH_DIRECT=0x15, IT_SET_SH_REG=0x76, IT_RELEASE_MEM=0x49)
- Kernel descriptors: projects/rocr-runtime/runtime/hsa-runtime/inc/amd_hsa_kernel_code.h
  (amd_kernel_code_t with compute_pgm_rsrc1/2, kernel_code_properties)

### What the experiments should do:

Phase 0 - Baseline Measurement (no code changes):
1. Write a benchmark that measures empty kernel dispatch latency for:
   (a) Eager hipLaunchKernel (N=10000 iterations)
   (b) Graph hipGraphLaunch with 1 kernel
   (c) Graph hipGraphLaunch with 100 kernels (measure per-kernel overhead)
2. Run with rocprof --hip-trace --hsa-trace to get GPU-side timing
3. The difference between (b) and (c) per-kernel shows the MEC overhead floor

Phase 1 - PM4 via vendor AQL (modify rocvirtual.cpp or standalone HSA benchmark):
1. Get kernel_object (GPU code address) and compute_pgm_rsrc1/2 from the kernel
   descriptor for a simple __global__ void empty_kernel() {}
2. Allocate kernarg segment via hsa_memory_allocate
3. Build PM4 command buffer:
   - SET_SH_REG: COMPUTE_NUM_THREAD_X/Y/Z = 1,1,1
   - SET_SH_REG: COMPUTE_PGM_LO/HI = kernel_code_addr >> 8
   - SET_SH_REG: COMPUTE_PGM_RSRC1/2 from kernel descriptor
   - SET_SH_REG: COMPUTE_RESOURCE_LIMITS = 0x3ff
   - SET_SH_REG: COMPUTE_USER_DATA_0 = kernarg_addr (lo/hi)
   - DISPATCH_DIRECT: dim_x=1, dim_y=1, dim_z=1,
     dispatch_initiator = COMPUTE_SHADER_EN | FORCE_START_AT_000 | USE_THREAD_DIMENSIONS
   - RELEASE_MEM: signal completion
4. Submit via ExecutePM4 (need HSA API access) or via a new CLR VirtualGPU method
5. Measure latency vs AQL dispatch

Phase 2 - Raw PM4 queue (most invasive):
1. Create a PM4 compute queue via hsaKmtCreateQueueV2 with HSA_QUEUE_COMPUTE
2. Directly write PM4 commands to ring buffer
3. Ring doorbell (write to wptr, then doorbell MMIO)
4. Measure latency

### Important notes:
- Register offsets (mmCOMPUTE_*) differ by GFX generation. Check your GPU's ASIC ID
  first with rocminfo and use the matching gc_*_offset.h file.
- SET_SH_REG reg_offset is relative to 0x2c00 (the SH register base). So for
  COMPUTE_PGM_LO = 0x0e0c (GFX9), the SET_SH_REG offset = 0x0e0c - 0x0c00 = 0x020c.
  (Note: the base varies: 0x2c00 for PACKET3_SET_SH_REG_START in soc15d.h.
  The pm4_cmds.h uses 0x2E** notation which is a different base. Check carefully.)
- The dispatch_initiator bits: bit 0 = COMPUTE_SHADER_EN (must be 1),
  bit 2 = FORCE_START_AT_000, bit 5 = USE_THREAD_DIMENSIONS,
  bit 15 = CS_W32_EN (for wave32 mode, GFX10+).
- For the V3 kernel descriptor (modern GFX9+ code objects), the kernel code address
  is: descriptor_address + (amdhsa::kernel_descriptor_t).kernel_code_entry_byte_offset.
  compute_pgm_rsrc1/2 are directly in the descriptor struct.

Please start by:
1. Running `rocminfo` to identify the GPU and its GFX version
2. Building and running the Phase 0 baseline benchmark
3. Examining the results to quantify the MEC overhead
4. Then proceeding to Phase 1 if the overhead looks significant
```

---

## Environment Setup Checklist

Before running experiments on the GPU machine:

```bash
# 1. Check ROCm installation
rocminfo | head -40
hipcc --version

# 2. Identify GPU generation (needed for register offsets)
rocminfo | grep -i "Name\|gfx"
# Output like: "gfx90a" (MI200), "gfx942" (MI300), "gfx1100" (7900XT), etc.

# 3. Clone this monorepo (if not already present)
git clone <repo-url> rocm-systems
cd rocm-systems

# 4. Build CLR (HIP runtime) - needed if modifying dispatch code
cd projects/clr
mkdir build && cd build
cmake -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd \
      -DCMAKE_PREFIX_PATH=/opt/rocm/ \
      -DCMAKE_INSTALL_PREFIX=/opt/rocm/ ..
make -j$(nproc)

# 5. Build ROCr (HSA runtime) - needed if modifying ExecutePM4 or queue creation
cd ../../rocr-runtime
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm/ ..
make -j$(nproc)

# 6. Build HIP graph perf tests (baseline)
cd ../../hip-tests/catch
mkdir build && cd build
cmake .. -DHIP_PLATFORM=amd -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
      -DOFFLOAD_ARCH_STR="--offload-arch=$(rocminfo | grep gfx | head -1 | awk '{print $2}')" \
      -DBUILD_PERF_TESTS=ON
make -j$(nproc) build_tests

# 7. Build the standalone benchmark
/opt/rocm/bin/hipcc benchmark_dispatch_latency.cpp -o benchmark_dispatch_latency

# 8. Run baseline
./benchmark_dispatch_latency
rocprof --hip-trace --hsa-trace ./benchmark_dispatch_latency
```

---

## Summary of Expected Results

| Measurement | Expected Value | Notes |
|---|---|---|
| Eager empty kernel | ~3-8 μs | Host overhead + MEC + doorbell |
| Graph 1-kernel | ~2-5 μs | Less host overhead, same MEC |
| Graph 100-kernel batch | ~1-2 μs/kernel | Host overhead amortized, MEC is now dominant |
| PM4 via vendor AQL | ~0.5-1.5 μs/kernel | Saves MEC AQL→PM4 translation |
| Raw PM4 queue | ~0.3-1 μs/kernel | Saves everything; similar to tinygrad |

The gap between "Graph 100-kernel batch" and "PM4 via vendor AQL" is the MEC overhead
we're trying to measure and eliminate.
