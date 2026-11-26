# 예전 버전(그라디오) 노트북 → HTML 웹앱 내보내기 (2026)

# ---------- 웹앱 기본 틀 (선생님 코드 teacher("W2")가 씀) ----------
WEB_CSS = """
*{box-sizing:border-box}body{margin:0;background:#FFFBF1;color:#2B1B22;font-family:system-ui,-apple-system,'Noto Sans KR',sans-serif;line-height:1.6}
main{max-width:560px;margin:0 auto;padding:28px 18px 48px}h1{font-size:26px;margin:0 0 6px}.sub{color:#5B3A45;margin:0 0 18px}
textarea{width:100%;font:inherit;font-size:16px;padding:12px;border:2px solid #6B1A35;border-radius:14px;background:#fff}
button,label.btn{display:inline-flex;align-items:center;justify-content:center;margin-top:10px;font:inherit;font-weight:700;font-size:16px;padding:10px 20px;min-height:46px;border-radius:999px;border:2px solid #6B1A35;background:#6B1A35;color:#FFFBF1;cursor:pointer}
.ex{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.ex button{background:#fff;color:#6B1A35;font-size:13px;min-height:36px;padding:6px 12px;margin:0}
#out{margin-top:20px;background:#fff;border:2px solid #6B1A35;border-radius:18px;padding:18px}#top{margin:0;font-size:24px}#info{margin:6px 0 12px}
.bar{display:grid;grid-template-columns:7em 1fr 3.5em;gap:8px;align-items:center;font-size:14px;margin:4px 0}.bar i{display:block;height:12px;border-radius:6px;background:#E9873A}
.track{background:#F1E7DA;border-radius:6px}.foot{color:#5B3A45;font-size:13px;margin-top:16px}#preview{max-width:100%;border-radius:14px;margin-top:12px;display:none}
#selftest.okk{color:#2E7D32;font-weight:700}#selftest.bad{color:#B3261E;font-weight:700}
"""

TEXT_HTML = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI 웹앱</title><style>""" + WEB_CSS + """</style></head>
<body><main>
<h1 id="title">AI 웹앱</h1>
<p class="sub">글을 넣으면 종류를 알려 주는 연습용 AI예요. 개인정보는 넣지 마세요.</p>
<label for="msg" class="sub">글</label>
<textarea id="msg" rows="5" placeholder="여기에 글을 넣으세요"></textarea>
<div class="ex" id="ex"></div>
<button id="go" type="button">AI에게 물어보기</button>
<section id="out" hidden aria-live="polite"><h2 id="top"></h2><p id="info"></p><div id="bars"></div></section>
<p class="foot">연습용 AI라 틀릴 수 있어요. 넣은 글은 이 화면 밖으로 보내지 않아요.</p>
<p class="foot" id="selftest">모델을 불러오는 중…</p>
</main>
<script>
let M = null;
function grams(text, lo, hi) {
  const out = [];
  for (const raw of text.toLowerCase().split(/\\s+/).filter(Boolean)) {
    const w = Array.from(' ' + raw + ' ');
    for (let n = lo; n <= hi; n++) {
      let off = 0; out.push(w.slice(off, off + n).join(''));
      while (off + n < w.length) { off++; out.push(w.slice(off, off + n).join('')); }
      if (off === 0) break;
    }
  }
  return out;
}
function predict(text) {
  const t = M.tfidf, cnt = new Map();
  for (const g of grams(text, t.ngram_range[0], t.ngram_range[1])) {
    const i = t.vocabulary[g]; if (i !== undefined) cnt.set(i, (cnt.get(i) || 0) + 1);
  }
  let idx = [], val = [], s = 0;
  for (const [i, c] of cnt) { const v = (t.sublinear_tf ? 1 + Math.log(c) : c) * t.idf[i]; idx.push(i); val.push(v); s += v * v; }
  s = Math.sqrt(s) || 1;
  let L0 = M.layers[0], h = L0.bias.slice();
  idx.forEach((i, k) => { const row = L0.weights[i], x = val[k] / s; for (let j = 0; j < h.length; j++) h[j] += x * row[j]; });
  h = act(h, L0.activation);
  for (let l = 1; l < M.layers.length; l++) {
    const L = M.layers[l], o = L.bias.slice();
    for (let i = 0; i < h.length; i++) { const r = L.weights[i]; if (h[i]) for (let j = 0; j < o.length; j++) o[j] += h[i] * r[j]; }
    h = act(o, L.activation);
  }
  return h;
}
function act(v, a) {
  if (a === 'relu') return v.map(x => Math.max(0, x));
  if (a === 'sigmoid') return v.map(x => 1 / (1 + Math.exp(-x)));
  if (a === 'softmax') { const m = Math.max(...v), e = v.map(x => Math.exp(x - m)), s = e.reduce((p, q) => p + q, 0); return e.map(x => x / s); }
  return v;
}
function show(p) {
  const order = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]), top = order[0][1], name = M.classes[top];
  document.getElementById('top').textContent = name + ' ' + Math.round(p[top] * 100) + '%';
  document.getElementById('info').textContent = (M.info || {})[name] || '';
  const bars = document.getElementById('bars'); bars.innerHTML = '';
  for (const [v, i] of order) {
    const r = document.createElement('div'); r.className = 'bar';
    r.innerHTML = '<span></span><span class="track"><i></i></span><span></span>';
    r.children[0].textContent = M.classes[i]; r.querySelector('i').style.width = (v * 100).toFixed(1) + '%';
    r.children[2].textContent = (v * 100).toFixed(1) + '%'; bars.appendChild(r);
  }
  document.getElementById('out').hidden = false;
}
document.getElementById('go').onclick = () => {
  const t = document.getElementById('msg').value.trim();
  if (!M) return;
  if (!t) { document.getElementById('top').textContent = '글을 입력하세요.'; document.getElementById('info').textContent = ''; document.getElementById('bars').innerHTML = ''; document.getElementById('out').hidden = false; return; }
  show(predict(t));
};
fetch('model.json').then(r => r.json()).then(j => {
  M = j; document.title = j.title; document.getElementById('title').textContent = j.title;
  const ex = document.getElementById('ex');
  (j.examples || []).forEach((t, k) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = '예시 ' + (k + 1); b.onclick = () => { document.getElementById('msg').value = t; }; ex.appendChild(b); });
  let pass = 0, tests = j.tests || [];
  for (const tt of tests) { const p = predict(tt.text); if (p.every((v, i) => Math.abs(v - tt.probs[i]) < 0.01)) pass++; }
  const st = document.getElementById('selftest');
  st.textContent = tests.length ? (pass === tests.length ? '검산 통과 (' + pass + '/' + tests.length + ')' : '검산 실패 (' + pass + '/' + tests.length + ')') : '';
  st.className = 'foot ' + (pass === tests.length ? 'okk' : 'bad');
}).catch(() => { document.getElementById('selftest').textContent = 'model.json을 읽지 못했어요. 코랩의 preview_webapp()으로 열어 보세요.'; });
</script></body></html>
"""

IMAGE_HTML = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI 웹앱</title><style>""" + WEB_CSS + """</style>
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"></script></head>
<body><main>
<h1 id="title">AI 웹앱</h1>
<p class="sub">사진을 올리면 종류를 알려 주는 연습용 AI예요. 사람 얼굴은 올리지 마세요.</p>
<label class="btn" for="pic">사진 고르기 / 찍기</label>
<input id="pic" type="file" accept="image/*" hidden>
<img id="preview" alt="고른 사진">
<section id="out" hidden aria-live="polite"><h2 id="top"></h2><p id="info"></p><div id="bars"></div></section>
<p class="foot">연습용 AI라 틀릴 수 있어요. 사진은 이 화면 밖으로 보내지 않아요.</p>
<p class="foot" id="selftest">모델을 불러오는 중…</p>
</main>
<script type="application/json" id="cfg">__CONFIG__</script>
<script>
const C = JSON.parse(document.getElementById('cfg').textContent);
let S = null;
document.title = C.title; document.getElementById('title').textContent = C.title;
ort.InferenceSession.create('model.onnx').then(s => { S = s; document.getElementById('selftest').textContent = '준비됐어요. 사진을 골라 보세요.'; })
  .catch(e => { document.getElementById('selftest').textContent = 'model.onnx를 읽지 못했어요: ' + e; });
function show(p) {
  const order = Array.from(p).map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]), top = order[0][1], name = C.classes[top];
  document.getElementById('top').textContent = name + ' ' + Math.round(p[top] * 100) + '%';
  document.getElementById('info').textContent = (C.info || {})[name] || '';
  const bars = document.getElementById('bars'); bars.innerHTML = '';
  for (const [v, i] of order) {
    const r = document.createElement('div'); r.className = 'bar';
    r.innerHTML = '<span></span><span class="track"><i></i></span><span></span>';
    r.children[0].textContent = C.classes[i]; r.querySelector('i').style.width = (v * 100).toFixed(1) + '%';
    r.children[2].textContent = (v * 100).toFixed(1) + '%'; bars.appendChild(r);
  }
  document.getElementById('out').hidden = false;
}
async function run(img) {
  const [H, W] = C.size, cv = document.createElement('canvas'); cv.width = W; cv.height = H;
  const cx = cv.getContext('2d'); cx.drawImage(img, 0, 0, W, H);
  const d = cx.getImageData(0, 0, W, H).data, x = new Float32Array(H * W * 3);
  for (let i = 0, j = 0; i < d.length; i += 4) { x[j++] = d[i]; x[j++] = d[i + 1]; x[j++] = d[i + 2]; }
  const feeds = {}; feeds[S.inputNames[0]] = new ort.Tensor('float32', x, [1, H, W, 3]);
  const out = await S.run(feeds); return out[S.outputNames[0]].data;
}
document.getElementById('pic').onchange = e => {
  const f = e.target.files[0]; if (!f || !S) return;
  const img = document.getElementById('preview'); img.onload = async () => { img.style.display = 'block'; show(await run(img)); };
  img.src = URL.createObjectURL(f);
};
window.runOnImage = run;
</script></body></html>
"""

# ===== 웹앱 내보내기 (예전 버전 노트북용) =====
# 노트북 맨 아래에서  MY_ID = "2-1-07"  을 정한 뒤  %run -i export_webapp.py  로 실행해요.
# 1단계와 신경망(model)이 있으면 됩니다. 결과: webapp/ 폴더와 반-번호.zip
import os, re, sys, json, shutil, subprocess
import numpy as np
G = globals()

def _x_need(pkg, mod):
    try:
        return __import__(mod)
    except ImportError:
        print(f"... {pkg} 설치 중"); subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
        return __import__(mod)

def _x_stop(msg):
    print("\n[멈춤] " + msg)
    raise SystemExit

SAFE_TESTS = ["오늘 날씨가 맑아서 기분이 좋아요", "좋은 기사 잘 읽었습니다", "다음 경기도 응원할게요",
              "정책에 대한 의견은 서로 다를 수 있어요", "유익한 정보 감사합니다"]
WEB_DIR = "webapp"

def _x_grams(text, lo, hi):
    out = []
    for w in str(text).lower().split():
        w = " " + w + " "
        for n in range(lo, hi + 1):
            off = 0; out.append(w[off:off + n])
            while off + n < len(w):
                off += 1; out.append(w[off:off + n])
            if off == 0:
                break
    return out

def _x_json_predict(J, text):
    t = J["tfidf"]; voc = t["vocabulary"]; idf = t["idf"]; lo, hi = t["ngram_range"]
    cnt = {}
    for g in _x_grams(text, lo, hi):
        if g in voc:
            cnt[voc[g]] = cnt.get(voc[g], 0) + 1
    x = np.zeros(len(idf))
    for i, c in cnt.items():
        x[i] = ((1 + np.log(c)) if t.get("sublinear_tf") else c) * idf[i]
    n = np.linalg.norm(x); h = x / n if n > 0 else x
    for L in J["layers"]:
        h = h @ np.asarray(L["weights"], dtype="float64") + np.asarray(L["bias"], dtype="float64")
        a = L.get("activation", "linear")
        if a == "relu":
            h = np.maximum(h, 0)
        elif a == "softmax":
            e = np.exp(h - h.max()); h = e / e.sum()
        elif a == "sigmoid":
            h = 1 / (1 + np.exp(-h))
    return h

def _x_info():
    return {c: str((G.get("INFO") or {}).get(c, "")) for c in G["class_names"]}

def _x_text():
    import keras
    from keras import layers
    from sklearn.feature_extraction.text import TfidfVectorizer
    v, m = G.get("vectorizer"), G.get("model")
    ok_vec = getattr(v, "analyzer", None) == "char_wb" and getattr(v, "tokenizer", None) is None and hasattr(v, "idf_")
    if not ok_vec:
        need = [k for k in ["texts_train", "y_train", "texts_test", "y_test"] if k not in G]
        if need:
            _x_stop(f"{need}가 없어요. 노트북을 위에서부터 다시 실행한 뒤(런타임 → 모두 실행) 이 칸을 실행하세요.")
        print("1단계가 형태소(또는 다른) 방식이라, 웹앱용 글자 조각 모델을 하나 더 학습해요. 보고서의 모델과 숫자는 그대로 써요.")
        units = [L.units for L in m.layers if isinstance(L, layers.Dense)][:-1] or [64]
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=3000, min_df=2, sublinear_tf=True)
        Xtr = v.fit_transform(G["texts_train"]).toarray().astype("float32")
        keras.utils.set_random_seed(42)
        m = keras.Sequential([layers.Input(shape=(Xtr.shape[1],))] + [layers.Dense(u, activation="relu") for u in units]
                             + [layers.Dense(len(G["class_names"]), activation="softmax")])
        m.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        m.fit(Xtr, np.asarray(G["y_train"]), epochs=100, batch_size=32, validation_split=0.2, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
        acc = float((m.predict(v.transform(G["texts_test"]).toarray().astype("float32"), verbose=0).argmax(1) == np.asarray(G["y_test"])).mean())
        print(f"웹앱용 모델 시험 정확도: {acc * 100:.1f}% (포스터에 '웹앱은 글자 조각 모델'이라고 적어 두면 좋아요)")
    lay = []
    for L in m.layers:
        if isinstance(L, layers.Dense):
            W, b = L.get_weights()
            lay.append({"weights": np.round(W.astype("float64"), 6).tolist(), "bias": np.round(b.astype("float64"), 6).tolist(),
                        "activation": L.get_config()["activation"]})
        elif not isinstance(L, (layers.Dropout, layers.InputLayer)):
            _x_stop(f"웹앱으로 옮길 수 없는 층이 있어요({L.__class__.__name__}). teacher('2')로 기준 신경망을 만든 뒤 3단계부터 다시 하세요.")
    safe = not G.get("SHOW_EXAMPLES", True)
    J = {"title": G.get("PROJECT_TITLE") or "My text AI", "classes": list(G["class_names"]), "info": _x_info(),
         "tfidf": {"analyzer": "char_wb", "ngram_range": list(v.ngram_range), "lowercase": True, "sublinear_tf": bool(v.sublinear_tf),
                   "norm": "l2", "vocabulary": {k: int(i) for k, i in v.vocabulary_.items()}, "idf": np.round(v.idf_.astype("float64"), 6).tolist()},
         "layers": lay, "examples": [] if safe else [str(t)[:120] for t in list(G.get("texts_test", []))[:3]]}
    # 검산: 시험 글로 파이썬과 비교 (화면 검산용 글은 안전 모드면 무해한 문장만 저장)
    chk = list(G.get("texts_test", []))[:5] or SAFE_TESTS
    ref = m.predict(np.asarray(v.transform(chk).toarray(), dtype="float32"), verbose=0)
    diff = max(float(np.max(np.abs(_x_json_predict(J, t) - ref[i]))) for i, t in enumerate(chk))
    if diff > 0.01:
        _x_stop(f"웹 계산이 파이썬과 달라요(차이 {diff:.4f}). 선생님께 알려 주세요.")
    tests = SAFE_TESTS if safe else chk
    J["tests"] = [{"text": str(t), "probs": [round(float(p), 6) for p in _x_json_predict(J, t)]} for t in tests]
    json.dump(J, open(f"{WEB_DIR}/model.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    open(f"{WEB_DIR}/index.html", "w", encoding="utf-8").write(TEXT_HTML)
    print(f"글 웹앱 완성: 파이썬과 차이 {diff:.1e}" + (" (안전 모드: 예시 칸 비움, 검산은 무해한 문장)" if safe else ""))

def _x_caffe(x):
    import keras
    return keras.ops.flip(x, axis=-1) - keras.ops.convert_to_tensor([103.939, 116.779, 123.68])

def _x_image():
    import keras
    from keras import layers
    from PIL import Image
    _x_need("tf2onnx", "tf2onnx"); ort = _x_need("onnxruntime", "onnxruntime")
    need = [k for k in ["model", "class_names", "paths_test", "X_train"] if k not in G]
    if need:
        _x_stop(f"{need}가 없어요. 노트북을 위에서부터 다시 실행한 뒤(런타임 → 모두 실행) 이 칸을 실행하세요.")
    base = G.get("base") if isinstance(G.get("base"), keras.Model) else None
    if base is None:
        n = G["X_train"].shape[1]
        for k, val in list(G.items()):
            if isinstance(val, keras.Model) and val is not G["model"]:
                try:
                    if int(val.output_shape[-1]) == n and len(val.input_shape) == 4:
                        base = val; break
                except Exception:
                    pass
    if base is None:
        _x_stop("1단계의 사전학습 모델(base)을 찾지 못했어요. teacher('1')을 실행하고 2~3단계, W1을 다시 한 뒤 이 칸을 실행하세요.")
    H, W = int(base.input_shape[1] or 224), int(base.input_shape[2] or 224)
    sample = [Image.open(p).convert("RGB") for p in list(G["paths_test"])[:3]]
    f = G.get("photo_features")
    if callable(f):
        ref = G["model"].predict(np.asarray(f(sample), dtype="float32"), verbose=0)
    else:
        ref = G["model"].predict(G["X_test"][:3], verbose=0)
    x = np.stack([np.asarray(im.resize((W, H)), dtype="float32") for im in sample])
    pre = [("-1~1", lambda t: layers.Rescaling(1 / 127.5, offset=-1)(t)), ("0~1", lambda t: layers.Rescaling(1 / 255.0)(t)),
           ("0~255", lambda t: t), ("caffe", lambda t: layers.Lambda(_x_caffe)(t))]
    for name, fn in pre:
        inp = keras.Input((H, W, 3)); full = keras.Model(inp, G["model"](base(fn(inp))))
        if np.max(np.abs(full.predict(x, verbose=0) - ref)) < 0.03:
            full.export(f"{WEB_DIR}/model.onnx", format="onnx", verbose=False)
            break
    else:
        _x_stop("1단계 전처리를 따라 하지 못했어요. teacher('1')을 실행하고 2~3단계, W1을 다시 한 뒤 이 칸을 실행하세요.")
    s = ort.InferenceSession(f"{WEB_DIR}/model.onnx", providers=["CPUExecutionProvider"])
    got = s.run(None, {s.get_inputs()[0].name: x})[0]
    diff = float(np.max(np.abs(got - ref)))
    if diff > 0.03:
        _x_stop(f"ONNX 결과가 파이썬과 달라요(차이 {diff:.4f}). 선생님께 알려 주세요.")
    mb = os.path.getsize(f"{WEB_DIR}/model.onnx") / 1e6
    if mb > 25:
        _x_stop(f"model.onnx가 {mb:.0f}MB라 25MB를 넘어요. 선생님께 알려 주세요(더 가벼운 BASE_MODEL로 다시 해야 해요).")
    cfg = {"title": G.get("PROJECT_TITLE") or "My photo AI", "classes": list(G["class_names"]), "info": _x_info(), "size": [H, W]}
    open(f"{WEB_DIR}/index.html", "w", encoding="utf-8").write(IMAGE_HTML.replace("__CONFIG__", json.dumps(cfg, ensure_ascii=False)))
    print(f"사진 웹앱 완성: 전처리 {name}, 입력 {H}x{W}, {mb:.1f}MB, 파이썬과 차이 {diff:.1e}")

def export_webapp(my_id):
    my_id = str(my_id).strip()
    if not re.fullmatch(r"\d-\d{1,2}-\d{2}", my_id):
        _x_stop('MY_ID는 "2-1-07"처럼 학년-반-번호(번호 두 자리)로 써요.')
    for k in ["model", "class_names"]:
        if k not in G:
            _x_stop(f"{k}가 없어요. 노트북을 위에서부터 다시 실행한 뒤(런타임 → 모두 실행) 이 칸을 실행하세요.")
    if "[" in str(G.get("PROJECT_TITLE", "")):
        _x_stop("설정 칸의 PROJECT_TITLE을 내 도구 이름으로 바꾼 뒤 설정 칸부터 다시 실행하세요.")
    shutil.rmtree(WEB_DIR, ignore_errors=True); os.makedirs(WEB_DIR)
    kind = "text" if "vectorizer" in G else "image"
    _x_text() if kind == "text" else _x_image()
    key = str(G.get("API_KEY", "") or "")
    for fn in os.listdir(WEB_DIR):
        if len(key) >= 6 and fn.endswith((".html", ".json")) and key in open(f"{WEB_DIR}/{fn}", encoding="utf-8").read():
            _x_stop("웹앱 파일에 API 인증키가 들어 있어요. 선생님께 알려 주세요.")
    if os.path.exists(my_id + ".zip"):
        os.remove(my_id + ".zip")
    shutil.make_archive(my_id, "zip", WEB_DIR)
    print(f"\n{my_id}.zip 완성 {sorted(os.listdir(WEB_DIR))} → 클래스룸 '웹앱 제출'에 그대로 내세요.")
    print("(선택) 화면을 꾸미고 싶으면 webapp/index.html을 고친 뒤 preview()로 보고 rezip(MY_ID)로 다시 묶어요.")
    try:
        import subprocess as _sp, time as _t
        from google.colab import output, files
        _sp.Popen([sys.executable, "-m", "http.server", "8000", "--directory", WEB_DIR], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        _t.sleep(1); print("아래 링크로 미리 열어 볼 수 있어요:"); output.serve_kernel_port_as_window(8000)
        files.download(my_id + ".zip")
    except Exception:
        pass


def rezip(my_id=None, folder=None):
    """webapp 폴더(꾸민 것, 제미나이로 직접 만든 것 모두)를 그대로 묶어요. 예: rezip("2-1-07")"""
    my_id = str(my_id or G.get("MY_ID", "")).strip()
    folder = folder or WEB_DIR
    if not re.fullmatch(r"\d-\d{1,2}-\d{2}", my_id):
        _x_stop('rezip("2-1-07")처럼 반-번호를 넣어요.')
    if not os.path.isfile(f"{folder}/index.html"):
        _x_stop(f"{folder} 폴더에 index.html이 있어야 해요. 지금: {sorted(os.listdir(folder)) if os.path.isdir(folder) else '폴더 없음'}")
    if os.path.exists(my_id + ".zip"):
        os.remove(my_id + ".zip")
    shutil.make_archive(my_id, "zip", folder)
    print(f"{my_id}.zip 묶기 완료 {sorted(os.listdir(folder))} → 클래스룸 '웹앱 제출'에 내세요.")
    try:
        from google.colab import files as _f
        _f.download(my_id + ".zip")
    except Exception:
        pass

def preview():
    """webapp 폴더를 새 창으로 미리 봐요."""
    try:
        import subprocess as _sp, time as _t
        from google.colab import output
        _sp.Popen([sys.executable, "-m", "http.server", "8000", "--directory", WEB_DIR], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        _t.sleep(1); output.serve_kernel_port_as_window(8000)
    except Exception as e:
        print("미리 보기를 열지 못했어요:", e)

if "MY_ID" in G:
    export_webapp(G["MY_ID"])
else:
    print('MY_ID = "2-1-07" 처럼 반-번호를 먼저 정하세요.')
