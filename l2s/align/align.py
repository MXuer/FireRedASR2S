from l2s.align.mms_funcs import load_model_dict, get_uroman_tokens, text_normalize, get_alignments, get_spans
from l2s.utils.model_registry import MMS_CODE_MAP
import logging

LOGGING_FORMAT = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
LOGGING_LEVEL = logging.INFO
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


class ALIGNER:

    def __init__(self, model_path, device) -> None:
        logging.info("Loading aligner model from {}".format(model_path))
        self.model, self.dictionary = load_model_dict(model_path)
        self.dictionary["<star>"] = len(self.dictionary)
        self.model.eval().to(device)
        self.uroman_path = "uroman/bin"
        self.device = device

    def align(self, transcripts, wav_file, names, use_star, language, raw_transcripts):
        language = MMS_CODE_MAP.get(language, 'eng')
        transcripts = [ele for ele in transcripts if ele.strip()]
        norm_transcripts = [text_normalize(line.strip(), language) for line in transcripts]
        tokens = get_uroman_tokens(norm_transcripts, self.uroman_path, language)
        if use_star:
            stars = ["<star>"] * len(tokens)
            tokens = [i for pair in zip(tokens, stars) for i in pair]
            tokens = ["<star>"] + tokens
            transcripts = [i for pair in zip(transcripts, stars) for i in pair]
            transcripts = ["<star>"] + transcripts
            norm_transcripts = [i for pair in zip(norm_transcripts, stars) for i in pair]
            norm_transcripts = ["<star>"] + norm_transcripts

        segments, stride = get_alignments(str(wav_file), tokens, self.model, self.dictionary, use_star, self.device)
        spans = get_spans(tokens, segments)
        align_segments = []
        true_index = 0
        for i, t in enumerate(transcripts):
            span = spans[i]
            if t == '<star>': continue
            seg_start_idx = span[0].start
            seg_end_idx = span[-1].end
            audio_start = round(seg_start_idx * stride / 1000, 3)
            audio_end = round(seg_end_idx * stride / 1000, 3)
            align_segments.append(
                {
                    "start": round(audio_start, 3),
                    "end": round(audio_end, 3),
                    "duration": round(audio_end - audio_start, 3),
                    "clean_text": t,
                    'text': raw_transcripts[true_index],
                    "name": names[true_index]
                }
            )
            true_index += 1
        return align_segments