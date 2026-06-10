"""커스텀 음성으로 공격 테스트.

사용법:
    # 폴더에 팀원 음성 파일 넣고 실행
    python -m voice_defense.scripts.run_custom_attack --voice-dir data/custom_voices

    # 한국어 텍스트로 복제하려면
    python -m voice_defense.scripts.run_custom_attack --voice-dir data/custom_voices --language ko

    # 특정 텍스트로 복제
    python -m voice_defense.scripts.run_custom_attack --voice-dir data/custom_voices \\
        --texts "안녕하세요 본인 확인 부탁드립니다" "오늘 날씨가 참 좋습니다"
"""

import argparse
import logging

from ..common.compat import apply_all_patches
apply_all_patches()

from ..attack.clone.clone_custom import clone_and_verify_custom

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="커스텀 음성 공격 테스트")
    parser.add_argument("--voice-dir", required=True,
                        help="음성 파일이 들어있는 폴더 (wav/flac/mp3)")
    parser.add_argument("--output-root", default="./outputs",
                        help="결과 저장 경로")
    parser.add_argument("--texts", nargs="*", default=None,
                        help="복제할 텍스트 (여러 개 가능)")
    parser.add_argument("--language", default="en",
                        help="합성 언어 (en, ko, ja, zh 등)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", nargs="*", type=float,
                        default=[0.15, 0.20, 0.25, 0.30, 0.35],
                        help="인증 임계값 목록")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("커스텀 음성 공격 테스트")
    logger.info(f"음성 폴더: {args.voice_dir}")
    logger.info(f"언어: {args.language}")
    logger.info("=" * 60)

    clone_result, scores_df = clone_and_verify_custom(
        voice_dir=args.voice_dir,
        output_root=args.output_root,
        texts=args.texts,
        threshold_sweep=args.threshold,
        language=args.language,
        device=args.device,
    )

    if scores_df is not None:
        logger.info(f"\n결과 저장 위치: {args.output_root}/custom_cloned/results/")


if __name__ == "__main__":
    main()
