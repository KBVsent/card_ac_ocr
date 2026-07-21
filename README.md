# CARD_AC_OCR

Vite + React 的静态网页应用，用于在浏览器中自动定位 Aime 卡片、校正方向与透视，并提取 20 位 ACCESS CODE。

## 环境要求

- Node.js 22.13 或更高版本
- npm 10 或更高版本
- 支持 WebAssembly、Canvas 和 Web Worker 的现代浏览器
- 如需网页直接拍照，必须通过 HTTPS 或 `localhost` 打开并允许摄像头权限

## 本地开发

```bash
npm ci
npm run dev
```

打开终端显示的本地地址。

## 类型检查与构建

```bash
npm run check
npm run build
```

## 本机预览生产构建

```bash
npm run preview
```

## 核心目录

```text
src/CardScanner.tsx  视觉检测、OCR 和界面状态
src/styles.css       页面样式
src/main.tsx         React 挂载入口
index.html           静态 HTML 入口与元数据
public/              图标和社交预览图片
```

## 隐私说明

- 照片和识别结果只存在于当前浏览器页面中。
- CDN 只下发 OpenCV 和 Tesseract 程序/模型，不接收照片。
- jsDelivr 仍会收到常规资源请求信息，例如 IP、时间和 Referer。

## 已知边界

- 严重遮挡、失焦、强反光或分辨率过低时可能失败。
- 20 位格式校验不代表每一位一定正确，使用前仍应人工核对。
