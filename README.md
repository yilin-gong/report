
# Web AI 助理 · 原型（Vite + React）

> 无登录、前端读取 CSV 的可运行原型。支持逾期清单、日报中心、AI 搜索台。

## 本地运行
```bash
cd web_ai_assistant_mvp
npm i
npm run dev
```

## 使用
1. 在页面顶部上传以下 CSV：`建议重点回访清单_医院级.csv`, `补全后续行动清单.csv`（可选：`hospital_data.csv`）
2. “逾期清单”支持筛选、批量生成“后续行动”、导出
3. “AI 搜索台”支持关键词检索（演示）

## 部署到 Vercel
- 已提供 `vercel.json`（Vite 构建 + SPA 回退），以及 `api/health.js` 健康检查函数。

### CLI 部署
```bash
npm i -g vercel
vercel
vercel --prod
```

### 控制台导入
- 推到 Git 仓库后在 Vercel 选择 Vite 框架，Build: `npm run build`，Output: `dist`
- 构建完成后访问 `/api/health` 应返回 `{ ok: true }`


---

## 设置 API & Key（前端）
- 右上角点击「设置」可填写：API 基地址（默认 /api）、（开发用）OpenAI Key、模型名。
- 为安全起见：生产环境应通过 Vercel 环境变量配置 `OPENAI_API_KEY`，不要在浏览器保存密钥。

## 安全建议
- 仅在**自测**时使用“从浏览器随请求发送 Key”的选项；否则请关闭并由后端持有 Key。
