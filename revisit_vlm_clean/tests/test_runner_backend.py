from revisit_vlm_clean.runner import BackendConfig, TGVFStage2Qwen3Backend, make_backend
from revisit_vlm_clean.schema import EvalMode, ForwardMode, RunConfig
from revisit_vlm_clean.stage2_runtime import Stage2RuntimeConfig


def test_make_tgvf_stage2_backend_without_prepare(tmp_path) -> None:
    runtime = Stage2RuntimeConfig(
        stage2_checkpoint=str(tmp_path / "ckpt.pt"),
        eval_jsonl=str(tmp_path / "eval.jsonl"),
    )
    config = RunConfig(
        run_id="stage2",
        checkpoint_path="outputs/checkpoint.pt",
        mode=EvalMode.TGVF_FORCE,
        post_tgvf_forward_mode=ForwardMode.KV_CACHE,
        subset_id="core_smoke_256_seed20260625",
    )
    backend = make_backend(BackendConfig(backend="tgvf_stage2_qwen3", stage2=runtime), config=config)

    assert isinstance(backend, TGVFStage2Qwen3Backend)
