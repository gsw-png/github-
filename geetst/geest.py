import hashlib
import json
import os
import random
import re
import requests
import time
import execjs
import cv2
import numpy as np

os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")
os.environ["EXECJS_RUNTIME"] = "Node"

# GeeTest v3: 312x160 乱序图按 52 块 10x80 还原成 260x160
RESTORE_INDEX = [
    39, 38, 48, 49, 41, 40, 46, 47, 35, 34, 50, 51, 33, 32, 28, 29,
    27, 26, 36, 37, 31, 30, 44, 45, 43, 42, 12, 13, 23, 22, 14, 15,
    21, 20, 8, 9, 25, 24, 6, 7, 3, 2, 0, 1, 11, 10, 4, 5, 19, 18, 16, 17,
]


def decode_image(content):
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("cv2.imdecode 失败, bytes=%s" % len(content or b""))
    return img


def restore_image(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    channels = img.shape[2]
    if w < 300 or h < 160:
        return img
    out = np.zeros((160, 260, channels), dtype=img.dtype)
    for idx, loc in enumerate(RESTORE_INDEX):
        src_x = loc % 26 * 12 + 1
        src_y = loc // 26 * 80
        dst_x = idx % 26 * 10
        dst_y = idx // 26 * 80
        out[dst_y:dst_y + 80, dst_x:dst_x + 10] = img[src_y:src_y + 80, src_x:src_x + 10]
    return out


def _to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return img[:, :, :3]
    return img


def make_track(distance):
    target = max(1, int(round(distance)))
    track = [[0, 0, 0]]
    x = 0
    y = 0
    t = random.randint(20, 60)
    while x < target:
        remain = target - x
        if remain > 50:
            step = random.randint(3, 7)
        elif remain > 15:
            step = random.randint(1, 3)
        else:
            step = 1
        x = min(target, x + step)
        if random.random() < 0.28:
            y += random.choice([-1, 0, 1])
            y = max(-3, min(3, y))
        t += random.randint(8, 16)
        track.append([x, y, t])
    t += random.randint(80, 180)
    track.append([target, y, t])
    return track


def recognize_gap(bg_url, slice_url, ypos=0, fullbg_url=None):
    """先还原乱序底图，再用 ypos 约束的模板匹配找缺口 x。"""
    scratch_dir = os.path.dirname(os.path.abspath(__file__))
    bg_resp = requests.get(bg_url, timeout=30)
    slice_resp = requests.get(slice_url, timeout=30)
    bg_raw = decode_image(bg_resp.content)
    slice_img = decode_image(slice_resp.content)
    bg = restore_image(bg_raw)

    bg_path = os.path.join(scratch_dir, "bg_restored.png")
    slice_path = os.path.join(scratch_dir, "slice.png")
    cv2.imwrite(bg_path, bg)
    cv2.imwrite(slice_path, slice_img)
    print("图片已保存:", bg_path, slice_path)
    print("背景图原始尺寸: %s -> 还原后 %s" % (bg_raw.shape, bg.shape))
    print("滑块图尺寸:", slice_img.shape, "ypos=", ypos)

    piece = slice_img
    mask = None
    y_off = 0
    if piece.ndim == 3 and piece.shape[2] == 4:
        piece_bgr = piece[:, :, :3]
        alpha = piece[:, :, 3]
        ys, xs = np.where(alpha > 20)
        if len(xs):
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            piece_bgr = piece_bgr[y0:y1, x0:x1]
            mask = alpha[y0:y1, x0:x1]
            y_off = y0
        else:
            piece_bgr = piece_bgr
            mask = alpha
    else:
        piece_bgr = _to_bgr(piece)

    search = _to_bgr(bg)
    search_y = 0
    th, tw = piece_bgr.shape[:2]
    bh, bw = search.shape[:2]
    ypos = int(ypos or 0) + y_off
    if 0 <= ypos <= bh - th:
        search = search[ypos:ypos + th, :]
        search_y = ypos

    if mask is not None:
        result = cv2.matchTemplate(search, piece_bgr, cv2.TM_CCORR_NORMED, mask=mask)
    else:
        result = cv2.matchTemplate(search, piece_bgr, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    tpl_x = int(max_loc[0])
    print("bg 模板匹配: x=%s y=%s score=%.4f" % (tpl_x, search_y, max_val))

    gap_x = tpl_x
    fullbg_tpl_x = None
    diff_x = None
    if fullbg_url:
        try:
            full_raw = decode_image(requests.get(fullbg_url, timeout=30).content)
            fullbg = restore_image(full_raw)
            full_search = _to_bgr(fullbg)
            if search_y:
                full_search = full_search[search_y:search_y + th, :]
            if mask is not None:
                full_res = cv2.matchTemplate(full_search, piece_bgr, cv2.TM_CCORR_NORMED, mask=mask)
            else:
                full_res = cv2.matchTemplate(full_search, piece_bgr, cv2.TM_CCOEFF_NORMED)
            _, full_score, _, full_loc = cv2.minMaxLoc(full_res)
            fullbg_tpl_x = int(full_loc[0])
            print("fullbg 模板匹配: x=%s score=%.4f" % (fullbg_tpl_x, full_score))

            diff = cv2.absdiff(_to_bgr(fullbg), _to_bgr(bg))
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            binary = cv2.medianBlur(binary, 5)
            colsum = binary.sum(axis=0)
            xs = np.where(colsum > binary.shape[0] * 8)[0]
            if len(xs):
                diff_x = int(xs[0])
                print("差分匹配: x=%s .. %s count=%s" % (diff_x, int(xs[-1]), len(xs)))
            # 完整图上的拼图位置才是落点；差分左缘常偏 3~6px。
            # 两者接近时用 fullbg 模板，否则退回差分。
            if fullbg_tpl_x is not None and diff_x is not None and abs(fullbg_tpl_x - diff_x) <= 10:
                gap_x = fullbg_tpl_x
                print("采用 fullbg 拼图位 x=%s（差分=%s bg模板=%s）" % (gap_x, diff_x, tpl_x))
            elif diff_x is not None:
                gap_x = diff_x
                print("采用差分左缘 x=%s（bg模板=%s fullbg模板=%s）" % (gap_x, tpl_x, fullbg_tpl_x))
        except Exception as exc:  # noqa: BLE001
            print("fullbg 匹配失败:", exc)

    gap_x = max(1, gap_x)
    print("最终识别缺口位置: x=%s" % gap_x)
    return gap_x, {"tpl": tpl_x, "fullbg_tpl": fullbg_tpl_x, "diff": diff_x}


def parse_jsonp(text):
    payload = (text or "").strip()
    if payload.startswith("{") or payload.startswith("["):
        return json.loads(payload)
    match = re.search(r"^[^(]+\((.*)\)\s*;?\s*$", payload, re.S)
    if not match:
        raise RuntimeError("无法解析 JSONP: %r" % payload[:200])
    return json.loads(match.group(1))


def jsonp_data(parsed):
    if not isinstance(parsed, dict):
        return {}
    data = parsed.get("data")
    return data if isinstance(data, dict) else parsed

js_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "极验.js")
with open(js_path, "r", encoding="utf-8") as f:
    js_code = f.read()
gct_path = "/Users/dxy/Downloads/fuck_js-master/cases/get-encrypt-cases/geetest.com/2026-10-08/gct.b71a9027509bc6bcfef9fc6a196424f5.js"
with open(gct_path, "r", encoding="utf-8") as f:
    js_code += "\n" + f.read()

timestamp_ms = int(time.time() * 1000)
session = requests.session()

headers = {
    "accept": "application/json, text/javascript, */*; q=0.01",
    "accept-language": "zh-CN,zh;q=0.9",
    "priority": "u=1, i",
    "referer": "https://demos.geetest.com/slide-float.html",
    "sec-ch-ua": "\"Google Chrome\";v=\"155\", \"Chromium\";v=\"155\", \"Not(A:Brand\";v=\"24\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"macOS\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36",
    "x-requested-with": "XMLHttpRequest"
}

#第1次获取数据包register-slide
url = 'https://demos.geetest.com/gt/register-slide'
params = {
    "t": timestamp_ms
}
response = session.get(url, headers=headers, params=params).json()
challenge = response['challenge']
gt = response['gt']

print("第一次获取数据包register-slide:\n",response)

#第2次获取数据包gettype.php
timestamp_ms = str(int(time.time() * 1000))
url = 'https://apiv6.geetest.com/gettype.php'
params = {
    "gt": gt,
    "callback": "geetest_"+timestamp_ms
}
response = session.get(url, headers=headers, params=params).text

print("第2次获取数据包gettype.php:\n",response)

#第3次获取数据包fullpage.js,不需要抓包

#第4次获取数据包get.php
params_w1 = execjs.compile(js_code).call('get_w1', gt, challenge)
timestamp_ms = str(int(time.time() * 1000))

api_headers = dict(headers)
api_headers.update({
    "accept": "*/*",
    "sec-fetch-dest": "script",
    "sec-fetch-mode": "no-cors",
    "sec-fetch-site": "cross-site",
})

url = "https://apiv6.geetest.com/get.php"
params = {
    "gt": gt,
    "challenge": challenge,
    "lang": "zh-cn",
    "pt": "0",
    "client_type": "web",
    "w": params_w1,
    "callback": "geetest_" + timestamp_ms,
}
response = session.get(url, headers=api_headers, params=params)

print("第4次获取数据包get.php:\n", response.text)
print(response)

#第5次获取数据包ajax.php
# 先打通空 w 通路，再取 slide3 配置，用 get_w2 生成最终 w。
timestamp_ms = str(int(time.time() * 1000))
url = "https://api.geevisit.com/ajax.php"
params = {
    "gt": gt,
    "challenge": challenge,
    "lang": "zh-cn",
    "pt": "0",
    "client_type": "web",
    "w": "",
    "callback": "geetest_" + timestamp_ms,
}
response = session.get(url, headers=api_headers, params=params)

print("第5次获取数据包ajax.php(空w通路):\n", response.text)
print(response)

# 取 slide3 拼图配置，拿到 c / s 后再打包 w2
timestamp_ms = str(int(time.time() * 1000))
url = "https://api.geevisit.com/get.php"
params = {
    "is_next": "true",
    "type": "slide3",
    "gt": gt,
    "challenge": challenge,
    "lang": "zh-cn",
    "https": "true",
    "protocol": "https://",
    "offline": "false",
    "product": "embed",
    "api_server": "api.geevisit.com",
    "callback": "geetest_" + timestamp_ms,
}
slide_resp = session.get(url, headers=api_headers, params=params)
slide_data = jsonp_data(parse_jsonp(slide_resp.text))
print("slide3 get.php:\n", slide_resp.text)

# ★ 关键修复：更新 challenge 为 slide3 返回的新 challenge（末尾带随机后缀）
new_challenge = slide_data.get("challenge") or challenge
print("更新后 challenge:", new_challenge)

# ★ 关键修复：下载背景图和滑块图，识别缺口距离
bg_path = (slide_data.get("bg") or "").lstrip("/")
slice_path = (slide_data.get("slice") or "").lstrip("/")
fullbg_path = (slide_data.get("fullbg") or "").lstrip("/")
static_server = slide_data.get("static_servers", ["static.geetest.com/"])[0]
if not static_server.startswith("http"):
    static_server = "https://" + static_server.rstrip("/") + "/"
bg_url = static_server + bg_path
slice_url = static_server + slice_path
fullbg_url = static_server + fullbg_path if fullbg_path else None
print("背景图:", bg_url)
print("滑块图:", slice_url)
print("完整图:", fullbg_url)

distance, gap_debug = recognize_gap(bg_url, slice_url, ypos=slide_data.get("ypos") or 0, fullbg_url=fullbg_url)
print("缺口对照:", gap_debug)

# ★ 完整流程函数：每次重新获取新 challenge，测试指定距离
def run_full_flow(test_dist=None):
    global session, headers, api_headers, js_code
    timestamp_ms = int(time.time() * 1000)
    s = requests.session()
    
    # 1. register-slide
    url = 'https://demos.geetest.com/gt/register-slide'
    resp = s.get(url, headers=headers, params={"t": timestamp_ms}).json()
    ch = resp['challenge']
    gt_id = resp['gt']
    
    # 2. gettype.php
    timestamp_ms = str(int(time.time() * 1000))
    url = 'https://apiv6.geetest.com/gettype.php'
    s.get(url, headers=headers, params={"gt": gt_id, "callback": "geetest_"+timestamp_ms})
    
    # 3. get.php (w1)
    params_w1 = execjs.compile(js_code).call('get_w1', gt_id, ch)
    timestamp_ms = str(int(time.time() * 1000))
    api_h = dict(headers)
    api_h.update({"accept": "*/*", "sec-fetch-dest": "script", "sec-fetch-mode": "no-cors", "sec-fetch-site": "cross-site"})
    url = "https://apiv6.geetest.com/get.php"
    s.get(url, headers=api_h, params={
        "gt": gt_id, "challenge": ch, "lang": "zh-cn", "pt": "0", "client_type": "web",
        "w": params_w1, "callback": "geetest_" + timestamp_ms,
    })
    
    # 4. ajax.php 空w通路
    timestamp_ms = str(int(time.time() * 1000))
    url = "https://api.geevisit.com/ajax.php"
    s.get(url, headers=api_h, params={
        "gt": gt_id, "challenge": ch, "lang": "zh-cn", "pt": "0", "client_type": "web",
        "w": "", "callback": "geetest_" + timestamp_ms,
    })
    
    # 5. slide3 get.php
    timestamp_ms = str(int(time.time() * 1000))
    url = "https://api.geevisit.com/get.php"
    resp = s.get(url, headers=api_h, params={
        "is_next": "true", "type": "slide3", "gt": gt_id, "challenge": ch, "lang": "zh-cn",
        "https": "true", "protocol": "https://", "offline": "false", "product": "embed",
        "api_server": "api.geevisit.com", "callback": "geetest_" + timestamp_ms,
    })
    slide = jsonp_data(parse_jsonp(resp.text))
    new_ch = slide.get("challenge") or ch
    
    # 6. 识别距离
    bg_path = (slide.get("bg") or "").lstrip("/")
    slice_path = (slide.get("slice") or "").lstrip("/")
    fullbg_path = (slide.get("fullbg") or "").lstrip("/")
    static_server = slide.get("static_servers", ["static.geetest.com/"])[0]
    if not static_server.startswith("http"):
        static_server = "https://" + static_server.rstrip("/") + "/"
    bg_url = static_server + bg_path
    slice_url = static_server + slice_path
    fullbg_url = static_server + fullbg_path if fullbg_path else None
    
    if test_dist is None:
        dist, gap_dbg = recognize_gap(bg_url, slice_url, ypos=slide.get("ypos") or 0, fullbg_url=fullbg_url)
        test_dist = gap_dbg["fullbg_tpl"]
    
    # 7. 生成 w2 提交
    ctx = execjs.compile(js_code)
    track = ctx.call("make_track", test_dist)
    passtime = track[-1][2]
    rp = hashlib.md5((gt_id + new_ch[:32] + str(passtime)).encode("utf-8")).hexdigest()
    w2_extra = {
        "c": slide.get("c") or [12, 58, 98, 36, 43, 95, 62, 15, 12],
        "s": slide.get("s") or "2c514d54",
        "passtime": passtime,
        "rp": rp,
        "lang": "zh-cn",
        "track": track,
        "imgload": random.randint(80, 180),
    }
    params_w2 = ctx.call("get_w2", gt_id, new_ch, test_dist, w2_extra)
    
    timestamp_ms = str(int(time.time() * 1000))
    url = "https://api.geevisit.com/ajax.php"
    resp = s.get(url, headers=api_h, params={
        "gt": gt_id, "challenge": new_ch, "lang": "zh-cn", "pt": "0", "client_type": "web",
        "w": params_w2, "callback": "geetest_" + timestamp_ms,
    })
    return test_dist, parse_jsonp(resp.text)


# ★ 自动微调：用差分匹配的结果作为基准（312宽坐标系），附近 ±10 像素逐个试
print("=" * 50)
print("第一步：先识别大概距离...")
base_dist, first_result = run_full_flow()
print(f"初始距离: {base_dist}, 结果: {first_result}")

if first_result.get("success") == 1:
    print(f"★★★ 一次就成功了！距离: {base_dist}")
else:
    print("=" * 50)
    print("第二步：在差分匹配结果附近 ±10 像素范围内微调...")
    # 用差分匹配的中间值作为基准（大概 125 左右）
    base_dist = 125
    for delta in range(-10, 11):
        test_d = base_dist + delta
        if test_d < 10:
            continue
        try:
            dist, result = run_full_flow(test_d)
            print(f"distance={dist} -> {result}")
            if result.get("success") == 1:
                print(f"★★★ 找到正确距离: {dist}")
                break
            time.sleep(0.5)
        except Exception as e:
            print(f"distance={test_d} -> error: {e}")
            time.sleep(1)