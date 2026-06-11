"""커스텀 음성 파일로 복제 테스트.

팀원 목소리 등 직접 녹음한 wav/flac 파일을 넣어서
XTTS v2로 복제하고 화자 인증 우회를 시도합니다.

사용법:
    # 폴더 안에 wav 파일들 넣기
    data/custom_voices/
    ├── 홍길동.wav
    ├── 김철수.wav
    └── 이영희.wav

    # 실행
    python -m voice_defense.scripts.run_custom_attack --voice-dir data/custom_voices
"""

import logging
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from ..zoo.tts.xtts import XTTSAttack
from ...common.audio import load_audio, save_audio

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def find_voice_files(voice_dir):
    """지정 폴더에서 음성 파일 찾기."""
    voice_dir = Path(voice_dir)
    if not voice_dir.exists():
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {voice_dir}")

    files = []
    for f in sorted(voice_dir.iterdir()):
        if f.suffix.lower() in SUPPORTED_FORMATS:
            files.append(f)

    if not files:
        raise FileNotFoundError(
            f"{voice_dir}에 음성 파일이 없습니다.\n"
            f"지원 형식: {', '.join(SUPPORTED_FORMATS)}"
        )

    logger.info(f"{len(files)}개 음성 파일 발견: {[f.name for f in files]}")
    return files


def clone_custom_voices(
    voice_dir,
    output_root="./outputs",
    texts=None,
    tts_model="tts_models/multilingual/multi-dataset/xtts_v2",
    language="en",
    device="cuda",
):
    """커스텀 음성 파일들로 복제 실험 실행.

    Args:
        voice_dir: 음성 파일이 들어있는 폴더 경로.
                   파일 이름이 화자 이름으로 사용됨 (예: 홍길동.wav → 화자 "홍길동")
        output_root: 결과 저장 경로.
        texts: 복제 음성으로 합성할 텍스트 목록.
        tts_model: 사용할 TTS 모델.
        language: 합성 언어 (en/ko 등).
        device: cuda 또는 cpu.

    Returns:
        실험 결과 dict.
    """
    if texts is None:
        texts = [
            "Please verify my identity for account access.",
            "The quick brown fox jumps over the lazy dog.",
            "This is a voice authentication test sample.",
        ]

    voice_files = find_voice_files(voice_dir)
    output_dir = Path(output_root) / "custom_cloned"
    output_dir.mkdir(parents=True, exist_ok=True)

    # XTTS 로드
    logger.info(f"XTTS v2 모델 로딩: {tts_model}")
    generator = XTTSAttack(model_name=tts_model, language=language, device=device)

    cloned_files = []
    reference_samples = {}

    for voice_file in tqdm(voice_files, desc="Cloning custom voices"):
        speaker_name = voice_file.stem  # 파일 이름 = 화자 이름
        speaker_dir = output_dir / speaker_name
        speaker_dir.mkdir(parents=True, exist_ok=True)

        # 원본 음성 복사 (리샘플링)
        ref_path = speaker_dir / f"ref_{speaker_name}.wav"
        wav, sr = load_audio(str(voice_file), target_sr=16000)
        duration_sec = len(wav) / 16000

        if duration_sec < 3:
            logger.warning(f"[{speaker_name}] 음성이 {duration_sec:.1f}초로 너무 짧습니다 (최소 3초). 건너뜁니다.")
            continue

        save_audio(wav, str(ref_path), sr=16000)
        reference_samples[speaker_name] = str(ref_path)
        logger.info(f"[{speaker_name}] {duration_sec:.1f}초 음성 로드 완료")

        # 복제
        for i, text in enumerate(texts):
            out_path = speaker_dir / f"clone_{speaker_name}_text{i:02d}.wav"
            try:
                result = generator.generate(text=text, reference_wav=str(voice_file))
                sf.write(str(out_path), result.waveform, result.sample_rate)
                cloned_files.append({
                    "speaker_name": speaker_name,
                    "text_id": i,
                    "text": text,
                    "reference_path": str(ref_path),
                    "cloned_path": str(out_path),
                    "original_duration": duration_sec,
                })
                logger.info(f"  → clone_text{i:02d}.wav 생성 완료")
            except Exception as e:
                logger.error(f"  → text{i:02d} 복제 실패: {e}")

    logger.info(f"\n총 {len(cloned_files)}개 복제 음성 생성 완료")
    logger.info(f"결과 저장 위치: {output_dir}")

    return {
        "voice_dir": str(voice_dir),
        "speakers": list(reference_samples.keys()),
        "reference_samples": reference_samples,
        "cloned_files": cloned_files,
        "texts": texts,
    }


def clone_and_verify_custom(
    voice_dir,
    output_root="./outputs",
    texts=None,
    threshold_sweep=None,
    tts_model="tts_models/multilingual/multi-dataset/xtts_v2",
    language="en",
    device="cuda",
):
    """커스텀 음성 복제 + 화자 인증 우회 테스트까지 한번에.

    Returns:
        (clone_result, scores_df) 튜플.
    """
    import pandas as pd
    from ..verify.speaker_verifier import SpeakerVerifier

    if threshold_sweep is None:
        threshold_sweep = [0.15, 0.20, 0.25, 0.30, 0.35]

    # 1. 복제
    clone_result = clone_custom_voices(
        voice_dir, output_root, texts, tts_model, language, device,
    )

    if not clone_result["cloned_files"]:
        logger.error("복제된 파일이 없습니다.")
        return clone_result, None

    # 2. 화자 인증
    logger.info("화자 인증 테스트 시작...")
    verifier = SpeakerVerifier(device=device)

    results = []
    for item in tqdm(clone_result["cloned_files"], desc="Verifying"):
        ref_path = item["reference_path"]
        clone_path = item["cloned_path"]
        score = verifier.similarity(ref_path, clone_path)

        for threshold in threshold_sweep:
            results.append({
                "speaker_name": item["speaker_name"],
                "text_id": item["text_id"],
                "text": item["text"],
                "threshold": threshold,
                "similarity_score": score,
                "verified": score >= threshold,
            })

    df = pd.DataFrame(results)

    # 결과 저장
    results_dir = Path(output_root) / "custom_cloned" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "custom_attack_scores.csv", index=False)

    # 요약 출력
    logger.info("\n" + "=" * 60)
    logger.info("커스텀 음성 공격 결과")
    logger.info("=" * 60)

    for name in clone_result["speakers"]:
        subset = df[df["speaker_name"] == name]
        for t in threshold_sweep:
            t_sub = subset[subset["threshold"] == t]
            asr = t_sub["verified"].mean()
            avg_score = t_sub["similarity_score"].mean()
            logger.info(f"  [{name}] threshold={t:.2f}: ASR={asr:.1%}, 평균유사도={avg_score:.3f}")

    logger.info("=" * 60)

    return clone_result, df
