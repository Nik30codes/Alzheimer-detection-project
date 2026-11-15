"""Local retrieval + constrained LLM evidence selection; never patient interpretation.

The small CPU model selects source IDs, not new medical claims. Only exact vetted
passages are rendered, with attribution. Failure falls back explicitly to retrieval.
"""
from functools import lru_cache
import json
from pathlib import Path
import re
import threading

from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'models/local_llm/qwen2.5-0.5b-instruct'
SOURCES = json.loads(Path(__file__).with_name('explanation_sources.json').read_text(encoding='utf-8'))
QUESTIONS = {
    'probability': 'What do AD and CN model probabilities mean? Can a model score diagnose disease?',
    'heatmap': 'What does Grad-CAM heatmap sensitivity mean? Is it damaged tissue or LIME?',
    'attention': 'How does learned slice attention combine features? Is it used in this website?',
    'augmentation': 'How is conditional DCGAN synthetic augmentation used in research training?',
    'evaluation': 'How are accuracy AUC thresholds and validation evaluated?',
    'batch': 'How are patient folder slices averaged? What happens with incomplete scans?',
    'privacy': 'Are patient images sent to an LLM service? How is data stored?',
}
_vectorizer = TfidfVectorizer(stop_words='english')
_matrix = _vectorizer.fit_transform([s['title'] + ' ' + s['text'] for s in SOURCES])
_lock = threading.Lock()
_model = _tokenizer = None


def retrieve(topic):
    query = _vectorizer.transform([QUESTIONS[topic]])
    scores = (_matrix @ query.T).toarray().ravel()
    return [SOURCES[i] for i in scores.argsort()[::-1][:3]]


def allowed_selection(text, sources):
    allowed = {s['id'] for s in sources}
    # Parse whole JSON only. Never render arbitrary model prose or invented sources.
    try:
        ids = json.loads(text.strip())
        if not isinstance(ids, list) or not 1 <= len(ids) <= 3: return None
        if any(not isinstance(i, str) or i not in allowed for i in ids): return None
        return list(dict.fromkeys(ids))
    except (ValueError, TypeError): return None


def selection_tokens(tokenizer, sources, prompt_length):
    """Constrain decoding to one complete, whitelisted JSON source selection."""
    candidates = [tokenizer.encode(json.dumps([s['id']]), add_special_tokens=False)
                  + [tokenizer.eos_token_id] for s in sources]
    def allowed(_batch_id, input_ids):
        suffix = input_ids.tolist()[prompt_length:]
        return sorted({c[len(suffix)] for c in candidates
                       if len(c) > len(suffix) and c[:len(suffix)] == suffix})
    return allowed


@lru_cache(maxsize=7)
def explain(topic):
    global _model, _tokenizer
    sources = retrieve(topic)
    mode = 'retrieval_only'
    status = 'Local model unavailable; showing retrieved source passages.'
    if (MODEL_DIR / 'model.safetensors').exists():
        if not _lock.acquire(blocking=False):
            return dict(mode='retrieval_only', status='Local model busy; showing retrieved source passages.', sources=sources)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if _model is None:
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
                _model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True,
                    trust_remote_code=False, dtype=torch.float32).eval()
            prompt = ('Select the single most relevant source ID for the question. Return ONLY a JSON array '
                      'containing that source ID. Do not answer the question or invent IDs.\n'
                      + QUESTIONS[topic] + '\n' + json.dumps(sources))
            text = _tokenizer.apply_chat_template([{'role':'user', 'content':prompt}], tokenize=False, add_generation_prompt=True)
            inputs = _tokenizer(text, return_tensors='pt')
            with torch.inference_mode():
                generated = _model.generate(**inputs, max_new_tokens=48, do_sample=False,
                    prefix_allowed_tokens_fn=selection_tokens(_tokenizer, sources, inputs['input_ids'].shape[1]),
                    pad_token_id=_tokenizer.eos_token_id)
            answer = _tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            selected = allowed_selection(answer, sources)
            if selected:
                # Always retain the highest-ranked evidence so an LLM selection cannot
                # omit the primary limitation accompanying the requested topic.
                selected = list(dict.fromkeys([sources[0]['id'], *selected]))
                sources = [s for sid in selected for s in sources if s['id'] == sid]
                mode, status = 'local_llm_rag', 'Local Qwen selected evidence; passages below are quoted from the project knowledge base.'
            else:
                status = 'Local model returned an invalid source selection; showing retrieved passages instead.'
        except (ImportError, OSError, RuntimeError, ValueError):
            status = 'Local model could not generate a valid response; showing retrieved source passages.'
        finally: _lock.release()
    return dict(mode=mode, status=status, sources=sources, model='Qwen2.5-0.5B-Instruct (CPU, local)' if mode == 'local_llm_rag' else None)
