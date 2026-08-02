# HailoRT GenAI Server Fixes for Hailo-10H (SOC_ACCELERATOR)

These changes fix four bugs that prevent LLM inference from working on Hailo-10H devices (SOC_ACCELERATOR type, e.g. Hailo-10H M.2 on Raspberry Pi 5).

Base version: HailoRT v5.3.0 (`d503417`)

Previously based on v5.2.0 (`41a720b`). Re-verified against v5.3.0 (released 2026-04-05):
upstream still calls `create_infer_model(hef, model_name)`, still has
`WAIT_FOR_OPERATION_TIMEOUT = seconds(10)`, and still never calls `set_resource_guard()`
anywhere under `hailort_server/`. All three bugs are unfixed upstream.

## Bug 1: Wrong InferModel type for SOC_ACCELERATOR devices

**Symptom:** `HAILO_NOT_IMPLEMENTED` — server logs: *"Not supported. Did you try calling create_configure_params on H10? If so, use InferModel instead"*

**Root cause:** `LLMInferenceManager::create()` called `vdevice->create_infer_model(hef, model_name)` (the `Hef` overload). On SOC_ACCELERATOR devices, `VDeviceHrpcClient` does not override this overload — only `create_infer_model(MemoryView, string)` and `create_infer_model(string, string)`. The call falls through to the base class `VDevice::create_infer_model(Hef, string)`, which creates an `InferModelBase`. `InferModelBase::configure()` calls `get_default_streams_interface()`, which always returns `HAILO_NOT_IMPLEMENTED` on `VDeviceHrpcClient`.

**Fix:** Pass `hef_buffer` (`shared_ptr<Buffer>`) through the creation chain and call `vdevice->create_infer_model(MemoryView(*hef_buffer), model_name)` instead. This routes to the `VDeviceHrpcClient` override that correctly creates an `InferModelHrpcClient`.

**Files changed:**
- `hailort/hailort_server/genai/llm/llm_inference_manager.hpp` — added `std::shared_ptr<Buffer> hef_buffer` parameter to `create()`
- `hailort/hailort_server/genai/llm/llm_inference_manager.cpp` — use `MemoryView(*hef_buffer)` overload
- `hailort/hailort_server/genai/llm/llm_server.hpp` — added `hef_buffer` parameter to `create_inference_managers_future()`
- `hailort/hailort_server/genai/llm/llm_server.cpp` — thread `hef_buffer` through call chain
- `hailort/hailort_server/genai/vlm/vlm_server.cpp` — pass `hef_buffer` in VLM call site

## Bug 2: Server-side timeout too short for Hailo-10H model loading

**Symptom:** `HAILO_TIMEOUT` during model creation — server times out waiting for async resource creation.

**Root cause:** `WAIT_FOR_OPERATION_TIMEOUT` was 10 seconds. On Hailo-10H (SOC_ACCELERATOR), model loading involves transferring the entire HEF buffer (~2.3 GB for Qwen2.5-1.5B) over PCIe RPC to the device firmware, which takes 30-60 seconds on a Raspberry Pi 5.

**Fix:** Increased `WAIT_FOR_OPERATION_TIMEOUT` from 10 seconds to 120 seconds.

**Files changed:**
- `hailort/hailort_server/genai/utils.hpp`

## Bug 3: Use-after-free in TokenEmbedder (SIGSEGV)

**Symptom:** `SIGSEGV` during token generation, specifically in `LLMPreProcess::update_cache_from_embeddings()` → `memcpy`. Crash occurs ~15-40 seconds after model loading succeeds.

**Root cause:** The `TokenEmbedder` is created with a `MemoryView` pointing into the HEF buffer via `hef.get_external_resources(INPUT_EMB_BINARY)`. Internally, `TokenEmbedder` wraps this as an `Eigen::Map` (a non-owning pointer). The `TokenEmbedder::create(MemoryView, ...)` overload does not call `set_resource_guard()`, so nothing keeps the underlying buffer alive.

After `handle_create_llm_request()` returns, all `shared_ptr` references to the HEF buffer are released:
1. The local `hef_buffer_ptr` in `handle_create_llm_request()` is destroyed
2. The `std::future` holding the creation lambda (which captured `hef_buffer`) is destroyed
3. All `Hef` copies (which hold `shared_ptr<Impl>` → `m_hef_buffer`) are destroyed

The HEF buffer is freed, leaving the `TokenEmbedder`'s `Eigen::Map` as a dangling pointer. The next call to `tokens_to_embeddings()` returns `EmbeddingViewWrapper`s pointing to freed memory, and `update_cache_from_embeddings()` crashes in `memcpy`.

**Fix:** After creating the `TokenEmbedder`, call `m_token_embedder->set_resource_guard(hef_buffer)` to store a `shared_ptr<Buffer>` in the embedder, keeping the HEF buffer alive for its lifetime.

**Files changed:**
- `hailort/hailort_server/genai/llm/llm_server.hpp` — added `hef_buffer` parameter to `create_token_embedder_future()`
- `hailort/hailort_server/genai/llm/llm_server.cpp` — pass `hef_buffer`, call `set_resource_guard()`
- `hailort/hailort_server/genai/vlm/vlm_server.hpp` — same signature change for VLM override
- `hailort/hailort_server/genai/vlm/vlm_server.cpp` — same fix for VLM

**Note:** This bug affects all device types, not just Hailo-10H. It is a latent use-after-free that may appear to work on some platforms due to the freed memory not being immediately reclaimed.

## Bug 4: the HEF is duplicated once per inference model (OOM on 8 GB hosts)

**Symptom:** `hailort_server` is OOM-killed while loading a large LLM. With `Qwen3-1.7B-Instruct`
(2.9 GB HEF) on an 8 GB Raspberry Pi 5, host memory climbs steadily to ~3.4 GB during the HEF read,
then jumps to ~7.8 GB in about ten seconds and the kernel kills the process.

**Root cause:** the fix for Bug 1 routes through `create_infer_model(MemoryView, name)`, and that
overload calls `Hef::create(const MemoryView &)`, which does
`Buffer::create_shared(hef_buffer.data(), hef_buffer.size(), ...)` — a full copy of the HEF.
`LLMServer` creates two inference models (prefill and token-by-token) from the same buffer, so the
peak is three copies: the one the server read from disk plus one per `create_infer_model` call. At
2.9 GB each that is 8.6 GB, which does not fit in 8 GB.

There is already a non-copying `Hef::create(std::shared_ptr<Buffer>)`, but `VDeviceHrpcClient` never
overrides the matching `create_infer_model(std::shared_ptr<Buffer>, const std::string &)`, so that
path falls through to the base class and hits Bug 1 all over again.

**Fix:** override `create_infer_model(std::shared_ptr<Buffer>, const std::string &)` in
`VDeviceHrpcClient` — same body as the `MemoryView` overload, but it keeps the caller's buffer via
`Hef::create(hef_buffer)` instead of duplicating it — and call that from `LLMInferenceManager`.
Peak host memory for `Qwen3-1.7B-Instruct` drops from >7.8 GB (OOM) to 3.4 GB.

**Files changed:**
- `hailort/libhailort/src/vdevice/vdevice_hrpc_client.hpp` — declare the override (plus the matching
  `HAILO_NOT_IMPLEMENTED` stub on `VDeviceSocketBasedClient`, for consistency with its siblings)
- `hailort/libhailort/src/vdevice/vdevice_hrpc_client.cpp` — implement it
- `hailort/hailort_server/genai/llm/llm_inference_manager.cpp` — use it instead of the `MemoryView`
  overload

**Note:** like Bug 3, this is not Hailo-10H specific in principle — any HRPC-client device loading a
large HEF pays the same duplication. It is just that LLM-sized HEFs are where it becomes fatal.

## Build

```bash
cd /home/jordan/hailort/build
cmake --build . --target hailort_server -j2
sudo cp hailort_server /usr/local/bin/hailort_server
sudo systemctl restart hailort-server
```
