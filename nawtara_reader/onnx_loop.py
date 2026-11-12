"""Staff reader on onnxruntime: greedy decoding with an EMPTY context after step 0.

homr's own loop (homr/transformer/decoder_inference.py) hands the
decoder one context token at every later step and drifts on this model: do not use it.

Added here: the probability of the chosen class of the rhythm, pitch and lift branches at each step
(the confidence figures written to conf.json).
"""
import numpy as np
import onnxruntime as ort


def _top_prob(logits):
    """Probability of the largest logit (softmax of one float32 vector)."""
    v = logits.astype(np.float32)
    v = v - v.max()
    e = np.exp(v)
    return float(e.max() / e.sum())


class OnnxStaffReader:
    def __init__(self, enc_path, dec_path, cfg, threads=1):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.enc = ort.InferenceSession(str(enc_path), so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(str(dec_path), so, providers=["CPUExecutionProvider"])
        self.cfg = cfg
        self.enc_dt = np.float16 if "float16" in self.enc.get_inputs()[0].type else np.float32
        ctx = [i for i in self.dec.get_inputs() if i.name == "context"][0]
        self.dec_dt = np.float16 if "float16" in ctx.type else np.float32
        self.inv = {n: {v: k for k, v in d.items()} for n, d in dict(
            rhythm=cfg.rhythm_vocab, pitch=cfg.pitch_vocab, lift=cfg.lift_vocab,
            articulation=cfg.articulation_vocab, slur=cfg.slur_vocab, position=cfg.position_vocab).items()}
        self.out_names = ["out_rhythms", "out_pitchs", "out_lifts", "out_positions",
                          "out_articulations", "out_slurs", "attention"]
        self.kv_in = [f"cache_in{i}" for i in range(cfg.decoder_depth * 4)]
        self.kv_out = [f"cache_out{i}" for i in range(cfg.decoder_depth * 4)]

    def encode(self, x):
        return self.enc.run(None, {"input": x.astype(self.enc_dt)})[0].astype(self.dec_dt)

    def predict(self, x):
        """x: float32 [1, 1, H, W]. Returns (symbols, probs): symbols are
        (rhythm, pitch, lift, articulation, slur, position) names, probs the (rhythm, pitch, lift)
        probabilities of the chosen classes, one per symbol."""
        cfg = self.cfg
        context = self.encode(x)
        empty = context[:, :0]
        heads, hd = cfg.decoder_heads, cfg.decoder_dim // cfg.decoder_heads
        cache = [ort.OrtValue.ortvalue_from_numpy(np.zeros((1, heads, 0, hd), dtype=self.dec_dt))
                 for _ in self.kv_in]
        tok = dict(r=1, p=0, l=0, a=0, s=0)
        out, probs = [], []
        for step in range(cfg.max_seq_len):
            feed = {"rhythms": np.array([[tok["r"]]], np.int64), "pitchs": np.array([[tok["p"]]], np.int64),
                    "lifts": np.array([[tok["l"]]], np.int64),
                    "articulations": np.array([[tok["a"]]], np.int64),
                    "slurs": np.array([[tok["s"]]], np.int64),
                    "context": context if step == 0 else empty,
                    "cache_len": np.array([step], np.int64)}
            b = self.dec.io_binding()
            for k, v in feed.items():
                b.bind_cpu_input(k, v)
            for n, c in zip(self.kv_in, cache):
                b.bind_ortvalue_input(n, c)
            for n in self.out_names + self.kv_out:
                b.bind_output(n, "cpu")
            self.dec.run_with_iobinding(b)
            o = b.get_outputs()
            cache = o[7:]
            last = [o[i].numpy()[0, -1, :] for i in range(6)]
            am = [int(v.argmax()) for v in last]  # rhythm, pitch, lift, position, articulation, slur
            if am[0] == cfg.eos_token:
                break
            tok = dict(r=am[0], p=am[1], l=am[2], a=am[4], s=am[5])
            inv = self.inv
            out.append((inv["rhythm"][am[0]], inv["pitch"][am[1]], inv["lift"][am[2]],
                        inv["articulation"][am[4]], inv["slur"][am[5]], inv["position"][am[3]]))
            probs.append((_top_prob(last[0]), _top_prob(last[1]), _top_prob(last[2])))
        return out, probs
