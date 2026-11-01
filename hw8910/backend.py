"""One Transformers implementation shared by full evaluation and real serving."""
from pathlib import Path
from release import canonical, digest, verify_artifact

TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'vocab.json', 'merges.txt', 'added_tokens.json')


def hashes(root, names):
    return {name: digest((Path(root)/name).read_bytes()) for name in names if (Path(root)/name).is_file()}


def tokenizer_fingerprint(root):
    return digest(canonical(hashes(root, TOKENIZER_FILES)))


def template_fingerprint(root):
    root = Path(root)
    if (root/'chat_template.jinja').is_file():
        return digest((root/'chat_template.jinja').read_bytes())
    import json
    return digest(canonical(json.loads((root/'tokenizer_config.json').read_text())['chat_template']))


class TransformersBackend:
    def __init__(self, manifest, root):
        import importlib.metadata
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.manifest = manifest
        self.runtime = manifest['runtime']
        runtime = self.runtime
        if digest(Path(__file__).read_bytes()) != runtime['backend_sha256']:
            raise ValueError('Backend code changed; new evaluation required')
        for package, version in runtime['packages'].items():
            if importlib.metadata.version(package) != version:
                raise ValueError('Runtime package changed: ' + package)
        artifact = verify_artifact(manifest, root)
        base = Path(manifest['base_artifact_uri'])
        # Base weights are pinned independently even for LoRA releases.
        actual = hashes(base, manifest['base_file_hashes'])
        if actual != manifest['base_file_hashes']:
            raise ValueError('Base artifact changed')
        tokenizer_root = artifact if manifest['kind'] == 'adapter' else base
        if tokenizer_fingerprint(tokenizer_root) != runtime['tokenizer_sha256']:
            raise ValueError('Tokenizer changed')
        if template_fingerprint(tokenizer_root) != runtime['template_sha256']:
            raise ValueError('Chat template changed')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
                                                        dtype=getattr(torch, runtime['dtype']))
        if manifest['kind'] == 'adapter':
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, artifact, local_files_only=True)
        self.device = runtime['device']
        self.model = self.model.to(self.device).eval()
        self.loaded_id = manifest['model_id']
        self.runtime_sha256 = manifest['runtime_sha256']
        self.calls = 0

    def warmup(self):
        # Real compute finishes before readiness. Keep measured request count separate.
        self.generate('Что такое Android? A: ОС. B: браузер. Варианты: A, B.',
                      self.runtime['generation']['max_new_tokens'])
        self.calls = 0

    def generate(self, prompt, limit):
        import torch
        g = self.runtime['generation']
        if limit != g['max_new_tokens']:
            raise ValueError('Generation differs from evaluation')
        text = self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': g['system']}, {'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=g['enable_thinking'])
        batch = self.tokenizer(text, return_tensors='pt').to(self.device)
        length = batch['input_ids'].shape[1]
        if length > self.runtime['max_input_tokens']:
            raise ValueError('Input exceeds evaluated input token limit')
        self.calls += 1
        if g['do_sample']:
            torch.manual_seed(g['seed'])
        with torch.inference_mode():
            out = self.model.generate(**batch, max_new_tokens=limit, do_sample=g['do_sample'],
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        tokens = out[0, length:]
        return {'answer': self.tokenizer.decode(tokens, skip_special_tokens=True),
                'output_tokens': int(tokens.numel())}

