# ByteRush Cyber 终端开场动画

直接预览：

```bash
python terminal_intro.py --force-animate
```

Cyber 动画使用霓虹扫描大 Logo，并在下方显示真实启动进度。宽终端显示完整 Logo，窄终端
自动切换为紧凑版本。

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

show_intro()
```

动画在 CI、输出重定向、`TERM=dumb` 或设置 `NO_COLOR` 时会自动降级为静态帧。设置
`BYTERUSH_NO_INTRO=1` 可关闭自动动画；如果希望完全不输出，则在入口中按该环境变量跳过
`show_intro` 调用即可。
