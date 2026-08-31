# ByteRush 终端开场动画 samples

直接预览全部方案：

```bash
python terminal_intro.py --all --force-animate
```

单独预览：

```bash
python terminal_intro.py --style cyber --force-animate
python terminal_intro.py --style neural --force-animate
python terminal_intro.py --style warp --force-animate
python terminal_intro.py --style minimal --force-animate
```

## 方案定位

- `cyber`：霓虹扫描大 Logo，视觉冲击最强，适合比赛演示或录屏。
- `neural`：展示 `IDEA → CODE → TRAIN → EVAL → SELECT`，最能表达项目的自主研究流程。
- `warp`：曲速点火和进度条，短、快，适合高频本地启动。
- `minimal`：克制的单行控制台风格，适合正式运行和长日志。

## 接入真实入口

### 日志与动画同时显示（推荐）

用实时启动器替代原命令中的脚本名，其他参数保持不变：

```bash
python launch_with_intro.py \
  --config bfts_config_kuairand.yaml \
  --load_ideas ai_scientist/ideas/kuairand_ranking.json \
  --load_code \
  --idea_idx 0 \
  --skip_plots \
  --skip_writeup \
  --skip_review
```

真实日志会在动画上方正常滚动。进度不是虚构的计时器，而是根据项目已有输出推进；进入
`Running code` 后到达 100%，Logo 扫描动画会继续运行。需要同时保存不含 ANSI 动画的纯日志：

```bash
python launch_with_intro.py --intro-log-file /tmp/byterush.log [原有参数...]
```

当输出被重定向或不在交互式终端中，启动器会自动打印静态 Logo，并退化为普通日志输出。

### 只播放一次

在 `launch_scientist_bfts.py` 中解析完参数、开始加载模型之前调用：

```python
from terminal_intro import show_intro

show_intro("neural")
```

动画在 CI、输出重定向、`TERM=dumb` 或设置 `NO_COLOR` 时会自动降级为静态帧。设置
`BYTERUSH_NO_INTRO=1` 可关闭自动动画；如果希望完全不输出，则在入口中按该环境变量跳过
`show_intro` 调用即可。
