# 三维资源导入服务

第一轮使用受限的 **glTF 2.0 JSON（`.gltf`）** 作为通用交换格式。服务把几何、材质、网格、节点、场景转换为内部内容寻址资源，后续格式只需产出同一套对象和依赖边。

## 支持范围

- glTF 2.0 `.gltf`，内嵌 `data:application/octet-stream;base64` 或同批相对外部 buffer。
- 三角形图元；`POSITION`、可选 `NORMAL`、标量索引；非索引图元会按连续三点转换。
- 无贴图的 PBR `pbrMetallicRoughness` 材质因子、`alphaMode`、`doubleSided`。
- 节点 TRS 或 16 元素矩阵、嵌套层级和多图元材质区间。
- 外部引用必须解析到 manifest 所在素材批目录内；绝对路径、URL、`..` 逃逸均拒绝。
- 动画、蒙皮、变形目标、相机、灯光、贴图、扩展、GLB 等范围外特性直接整批失败，不静默丢数据。

## 内部模型

对象 ID 由规范化 JSON 的 SHA-256 生成：

- `geo_`：位置、三角形索引、可选法线。
- `mat_`：材质参数。
- `msh_`：几何引用和 `(material_id, face_offset, face_count)` 材质绑定。
- `ndo_`：4x4 行主序变换、网格引用、子节点 Merkle 式引用。
- `scn_`：根节点集合。
- `ast_`：一次导入产出的资源根。

相同内容复用同一对象文件。导入先完成解析和完整错误收集，再写入对象与索引；依赖缺失、损坏或越界时不生成索引，不发布半成品。对象文件和 `index.json` 都通过临时文件原子替换。

## 快速开始

```bash
python3 scripts/create_sample.py samples/robot
python3 -m resource3d.cli import --library build/library --manifest samples/robot/robot.gltf > build/asset-id.txt
python3 -m resource3d.cli export \
  --library build/library \
  --asset "$(cat build/asset-id.txt)" \
  --output build/inspection.json
python3 -m unittest discover -s tests
```

样例包含：两个实例共享的立方体几何、一个外部 `.bin` 金字塔、两种材质、父节点旋转以及嵌套平移/缩放。导出的 JSON 可直接检查所有面、材质引用和完整层级变换。
