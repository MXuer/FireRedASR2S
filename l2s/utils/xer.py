import re
import jiwer

def text_normalize(text, language):
    text = re.sub('[,?!.。？！，》《<>「」」]', '', text)
    if language == "ko_kr":
        text = re.sub(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]", lambda x:" "+x[0]+" ", text)
    elif  language == "zh_cn_msl":
        text = re.sub(r"[\u4e00-\u9fa5]", lambda x:" "+x[0]+" ", text)
    elif  language == "ct_hk":
        text = re.sub(r"[\u4e00-\u9fa5]", lambda x:" "+x[0]+" ", text)
    elif  language == "ct_gz":
        text = re.sub(r"[\u4e00-\u9fa5]", lambda x:" "+x[0]+" ", text)
    elif language == "ja_jp":
        text = re.sub(r"[\u3040-\u309f\u4E00-\u9FFF\u30a0-\u30ff]", lambda x:" "+x[0]+" ", text)
    return text



def cal_xer_batch(input, language):
    name2xers = {}
    for (name, ref, hyp) in input:
        ref_norm, hyp_norm = text_normalize(ref, language), text_normalize(hyp, language)
        xer = jiwer.wer(ref_norm, hyp_norm)
        name2xers[name] = [xer, ref, hyp]
    return name2xers