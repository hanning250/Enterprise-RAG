# 韩宁知识助手 · 前端

React + Vite + TypeScript，对接后端 `/api/v2/rag/query`。

## 启动

1. 后端（项目根目录）：

```powershell
.\.venv\Scripts\python.exe main.py
```

2. 前端：

```powershell
cd web
copy .env.example .env
# 编辑 .env：VITE_INTERNAL_AUTH=与后端 AUTH_TRUSTED_IDENTITY_SECRET 相同
npm run dev
```

浏览器打开 http://localhost:5173 。开发态通过 Vite 代理访问 `http://127.0.0.1:8765`。

## 说明

- 请求拦截器自动附加 `X-Internal-Auth`、`X-Request-Id`
- 右侧「身份」面板仅用于本地联调，生产应由 SSO/网关注入 JWT
- `VITE_API_BASE` 默认为空（走同源代理）；直连后端时可填 `http://127.0.0.1:8765`
