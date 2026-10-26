# 逐帧 EE 弹簧训练队列

运行 `python -u scripts/wait_passive_ee_spring.py`，等待GPU 0–3或4–7任意整组释放，只启动一组。每30秒检查，无计算进程、每卡显存低于1024MiB且利用率低于10%，连续三次满足才启动；两组同时空闲时优先0–3。读取已有passive实验launcher PID，存活的训练流水线仍保留其GPU，避免在验证/阶段交接时抢占。

入口 `scripts/train_passive_ee_spring_pipeline.sh`。任务 `G1/G1_passive_ee_spring`，teacher和student开启逐帧弹簧力，刚度80N/m、每手30N、hold零力。每卡12288环境，从头训练teacher1.2B、adapt100M、student400M；保留teacher质量门槛。数据仍是现有AMASS短片段。

输出 `outputs/passive-ee-spring/`：`wait.log`等待日志，`queue.json`状态，`launch.log`训练日志，`latest_pipeline.txt`实际运行目录。文件锁阻止多个等待器；一旦已启动，重复运行不会再次训练。训练失败记录failed，不自动重试。

本次已验证GPU分组选择、缺卡/占用判断、现有流水线预留及shell语法，并后台启动等待器。弹簧模式此前已完成施力逻辑检查，实际PPO运行验证需等待GPU释放。此脚本是本机轮询器，不是集群调度器；不能防止其他独立程序在最后一次检查后同时抢占显卡。
