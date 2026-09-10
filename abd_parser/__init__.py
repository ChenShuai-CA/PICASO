# abd_parser — ABD .txt/.spec 解析与结果标签提取（Stage4 §5.1 最小实现）
#
# 模块：
#   txt_reader  — .txt 读取（仅解析所需通道，100Hz 原始数据不降采样）
#   spec_reader — .spec 分段解析（触发器配置 / 车辆参数 / 同步模式）
#   params      — condition 目录名 + .spec → 场景参数表（E 空间的 v0 子集）
#   labels      — 附录 G 结果标签：T0 定位 / 碰撞多证据 / minTTC / t_AEB
#   extract_labels — 全量提取管线 → inventory/labels_v0.csv
