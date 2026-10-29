# 队员开发说明（TasteBridge）

给 Steph、香香和以后接手的人：想改代码、本地跑起来、改完上线，看这一份就够了。

- 在线 demo：https://tastebridge-xofy.onrender.com
- 代码仓库：https://github.com/cqu-jia/tastebridge
- 截止时间（伦敦时间）：**Nebius 10/30 17:00**，**Qloo 10/31 03:45**

> ⚠️ **最重要的一条：推送到 `main` 分支会自动部署到线上 demo。** 评委看到的就是 `main` 上的代码。
> 所有改动先在自己的分支上完成，本地测好后开 Pull Request（PR），由另一位队员确认再合并。不要直接推到 `main`。
> **截止之后不要再推任何改动**：很多比赛规定截止后不能修改作品，评委看的是截止时的版本。

---

## 1. 第一次准备（每人做一次）

### 1.1 拿到仓库的写权限
1. 注册 GitHub 账号，把用户名发给思佳。
2. 思佳在仓库页面 **Settings → Collaborators → Add people** 邀请你。
3. 你会收到一封 GitHub 邀请邮件，点 **Accept invitation**。

没有写权限也能 clone 下来本地跑，只是推不上去。

### 1.2 装好工具
| 工具 | Mac | Windows |
|---|---|---|
| Git | 终端里运行 `git --version`，没有的话会提示安装 | 装 [Git for Windows](https://git-scm.com/download/win) |
| Python 3.9 或更新 | Mac 一般自带，运行 `python3 --version` 看看 | 装 [Python](https://www.python.org/downloads/)，安装时勾选 "Add Python to PATH" |
| GitHub 登录 | 推荐装 [GitHub CLI](https://cli.github.com/)，然后运行 `gh auth login` | 同左 |

### 1.3 下载代码
```bash
git clone https://github.com/cqu-jia/tastebridge.git
cd tastebridge
```

### 1.4 建 Python 环境、装依赖
Mac / Linux：
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Windows（PowerShell）：
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
以后每次打开新终端，都要先运行 `source .venv/bin/activate`（Windows 是 `.venv\Scripts\Activate.ps1`），命令行前面出现 `(.venv)` 就对了。

### 1.5 配置 key（`.env` 文件）
```bash
cp .env.example .env
```
然后用编辑器打开 `.env`，填上三个值：

| 变量 | 去哪里拿 |
|---|---|
| `QLOO_API_KEY` | 找思佳**私聊**要（Qloo 发给团队的 hackathon key） |
| `NEBIUS_API_KEY` | 找思佳私聊要，或者自己在 Nebius Token Factory 的 **API keys** 页面新建 |
| `NEBIUS_MODEL` | 已经填好：`nvidia/nemotron-3-super-120b-a12b` |

**关于 key 的规矩：**
- `.env` 已经写进 `.gitignore`，git 不会提交它。**千万不要把 key 写进任何 `.py`、`.html`、`.md` 文件**，仓库是公开的，推上去全世界都能看到。
- 不要在群聊里明文发 key，用私聊。
- 万一 key 被推上去了，马上告诉思佳去作废、换新的。只删掉文件没用，git 历史里还留着。
- 不填 Nebius key 也能跑：网站会用模板文字代替 AI 生成的解释，推荐本身照常出。

### 1.6 本地跑起来
```bash
uvicorn app.main:app --reload
```
浏览器打开 http://localhost:8000 。`--reload` 的意思是改了代码会自动重启，刷新网页就能看到效果。按 `Ctrl+C` 停止。

快速自检（另开一个终端）：
```bash
curl http://localhost:8000/health
```
返回 `{"ok":true,"llm":true}` 说明 Qloo 和 Nebius 都配好了；`"llm":false` 说明 Nebius key 没填对。

---

## 2. 代码在哪里，想改什么去哪儿改

```
app/
  main.py      网站后端入口：/api/plan（单人）、/api/twin（两人），以及页面实际在用的 /stream 版本（边算边推送进度）；限流也在这里
  engine.py    推荐算法：把用户的喜好变成地点排序（核心）
  agent.py     AI agent：Nemotron 调用工具、写推荐理由；所有提示词都在这里
  qloo.py      调 Qloo API 的封装，还有"哪些地点不算休闲场所"的过滤名单
  geo.py       邮编 / 地名 → 经纬度（用免费的 postcodes.io）
static/
  index.html   整个前端页面（HTML + CSS + JS 都在这一个文件里；6 题交互和 Leaflet 地图也在这里）
scripts/
  list_models.py    列出 Nebius 上可用的模型
  deploy_render.py  第一次部署用的脚本，平时不用管
  evaluate.py       20 个虚拟用户的对比评测，结果写到 eval/（README 里的图就是它生成的）
eval/
  chart.svg / summary.md / results.json   评测的图、表和原始数据
render.yaml   Render 部署配置
```

| 想改的东西 | 去哪里 |
|---|---|
| 页面文字、按钮、颜色、布局、示例数据 | `static/index.html`（示例数据在 `DEMO` 那一段） |
| AI 写的推荐理由的语气、规则、禁用词 | `app/agent.py` 里的 `SYSTEM_PROMPT`、`TWIN_PROMPT`、`AI_CLICHES` |
| 不想要的地点类型（比如酒店、学校） | `app/qloo.py` 里的 `NON_LEISURE` |
| 要往下压的连锁店（Costa、Pret 等） | `app/engine.py` 里的 `CHAINS` 和 `CHAIN_FACTOR` |
| "你的口味"卡片里显示哪些领域 | `app/engine.py` 里的 `PROFILE_DOMAINS` |
| 页面上 agent 每一步显示的文字 | `app/engine.py` 里的 `emit(...)`，以及 `app/agent.py` 里的 `_narrate` |
| 限流（每人每分钟、每天的次数） | `app/main.py` 顶部的 `PER_IP_PER_MINUTE` 等常量 |
| 推荐算法参数（搜索半径、噪声阈值、近期喜好权重等） | `app/engine.py` 顶部的常量，每个都有注释 |
| 换 AI 模型 | `.env` 里的 `NEBIUS_MODEL`（线上的要在 Render 后台改，见第 5 节） |

推荐算法的思路（两路融合、去噪、去流行度偏差、归因、两人最小痛苦融合）写在 `app/engine.py` 文件开头的注释里，`README.md` 里也有一张表。

---

## 3. GitHub 是怎么协作的

- **仓库（repository）**：大家共享的项目文件和修改历史；本项目在 [cqu-jia/tastebridge](https://github.com/cqu-jia/tastebridge)。
- **`main` 分支**：目前准备给评委看的版本。合并到 `main` 会触发 Render 自动部署。
- **自己的分支（branch）**：每项改动单独开一条，互不影响线上。一个分支尽量只做一件事。
- **提交（commit）**：给一组改动存档，说明改了什么。`commit` 只保存在自己电脑上；`push` 才把它传到 GitHub。
- **Pull Request（PR）**：请队友检查并把你的分支合并到 `main` 的申请。PR 里写清改动、测试方法和截图；有评论就继续在原分支修改并 `push`，PR 会自动更新。
- **冲突（conflict）**：两个人改了同一处，Git 无法替大家决定保留哪版。最近首页就发生过这种情况。处理前要看懂两边改动，不能直接选“全部保留我的”。

### 开工前先说一声

在群里用一句话说明“我准备改什么、会动哪些文件”。特别是 `static/index.html` 集中了整个前端，若两个人同时改它，先分好页面区域或等对方的 PR 合并。把未完成的想法写在 PR 描述或群里，让接手的人知道。

## 4. 改代码的流程

### 每次改动：开分支 + PR

```bash
# 1. 先把最新代码拉下来
git switch main
git pull --ff-only origin main

# 2. 开一个新分支，名字写清楚改什么
git switch -c steph/onboarding-copy

# 3. 改代码，本地跑起来测一遍（见 1.6）

# 4. 提交
git status            # 先看有哪些文件变了
git add static/index.html
git diff --cached     # 检查准备提交的内容，没有 .env、key 或无关文件
git commit -m "Reword onboarding questions"

# 5. 推到 GitHub
git push -u origin steph/onboarding-copy
```

推完后终端会打印一个链接，或者打开仓库页面，会看到黄色提示条 **Compare & pull request**：
1. 点它，写一两句改了什么，点 **Create pull request**；
2. 在 PR 描述里写“改了什么、如何测试、是否影响线上”，界面改动附截图；在群里请另一位队员看一眼；
3. 审查者确认后再点 **Merge pull request** → **Confirm merge**；
4. 合并到 `main` 后，Render 会**自动部署**，2–5 分钟后线上生效。

合并后，本地开始下一项工作之前，先同步最新版：
```bash
git switch main
git pull --ff-only origin main
```

### 常见问题
- **PR 提示和 `main` 有冲突**：先在群里告诉改过同一文件的队友。确认两边要保留的内容后，在自己的分支合并最新版 `main`，逐处解决 `<<<<<<<`、`=======`、`>>>>>>>` 标记，再本地测试、提交并推送。拿不准就让队友一起看，不要强行覆盖。
- **`git push` 被拒绝，提示 rejected / fetch first**：先用 `git status` 和 `git fetch origin` 看当前分支及远端改动；不要用 `--force`。请改过同一分支的队友一起处理。
- **推送时要求登录**：运行 `gh auth login`，按提示用浏览器登录 GitHub。
- **不小心把 `.env` 加进了暂存区**：运行 `git restore --staged .env`，然后检查 `.gitignore` 里有没有 `.env` 这一行。
- **推送前再检查一次**：`git status` 看提交文件；浏览器走完你的功能，也试一遍已有的单人和双人入口。

---

## 5. 线上部署（Render）

- 平台：Render 免费版，服务名 `tastebridge`，机房在法兰克福。
- **自动部署**：`main` 分支有新提交，Render 就会自动重新构建并上线，大约 2–5 分钟。
- **看部署状态和日志**：Render 后台 → `tastebridge` 服务 → **Events** / **Logs**。只有思佳的账号能进；要看的话让她截图，或者把你加进 Render 的 workspace。
- **线上的 key 和模型名**：在 Render 后台 → **Environment** 里改，改完会自动重新部署。**线上不读仓库里的 `.env`**。
- **免费版会休眠**：15 分钟没人访问就休眠，下次打开要等约 50 秒。给评委看之前，或者演示之前，自己先打开一次。
- **部署失败了怎么办**：看 Logs 最后几行红字。最常见的是代码报错（本地先跑通再推，就能避免）。线上坏了又来不及修时，可以在 GitHub 上 revert 刚才那个提交，或者让思佳在 Render 后台点 **Rollback** 回到上一个正常版本。

---

## 6. 额度和费用

- Nebius 账户里有 $30 赠送额度，一次推荐大约花几美分，够用。
- 本地测试也会消耗同一份额度，正常测没问题，别写循环狂刷。
- Qloo 的 hackathon key 有频率限制，代码里已经做了自动重试和缓存；短时间内大量请求还是可能被限。
- 线上 demo 有限流：同一个人每分钟最多 6 次、每天 60 次，全站每天 600 次。测试时被提示"太快了"就是这个。
- 跑一次 `scripts/evaluate.py` 大约要 100 多次 Nemotron 和几百次 Qloo 调用，别反复跑。

---

## 7. 提交前检查清单

- [ ] 线上 demo 能打开，"My week" 和 "Taste Twin" 都能出结果
- [ ] 手机上打开也正常
- [ ] README 能说清楚项目是什么、怎么用 Qloo、怎么用 Nemotron/Nebius
- [ ] 仓库里没有 key（GitHub 仓库页面搜 `hack_` 和 `v1.`，应该搜不到）
- [ ] Devpost 上三个人都加为队友
- [ ] Nebius 那场 10/30 17:00 前交，Qloo 那场 10/31 03:45 前交
- [ ] 截止后不再推任何改动
