# 三维资源导入服务

第一轮使用受限的 **glTF 2.0 JSON（`.gltf`）** 作为通用交换格式。服务把几何、材质、网格、节点、场景转换为内部内容寻址资源，后续格式只需产出同一套对象和依赖边。

## 支持范围

- glTF 2.0 `.gltf`，内嵌 `data:application/octet-stream;base64` 或同批相对外部 buffer。
- 三角形图元；`POSITION`、可选 `NORMAL`、标量索引；非索引图元会按连续三点转换。
- PBR `pbrMetallicRoughness` 材质因子、`alphaMode`、`doubleSided`，以及贴图绑定：`baseColorTexture`、`metallicRoughnessTexture`、`normalTexture`、`occlusionTexture`、`emissiveTexture`；同字节图像无论被多少材质、何种用途引用都共享同一个 `tex_` 身份，用途（`base_color` / `metallic_roughness` / `normal` / `occlusion` / `emissive`）显式记录，颜色与数据贴图不会混淆。
- 外部 PNG/JPEG 图像（相对路径，批目录内）或 `data:` base64 内嵌图像，按魔数校验类型。
- 蒙皮网格：`JOINTS_0`/`WEIGHTS_0` 逐顶点属性、`skins` 的有序关节节点与 `inverseBindMatrices`；骨骼权重、关节层级和节点变换在转换后可用内部矩阵直接求值。
- 节点 TRS 或 16 元素矩阵、嵌套层级和多图元材质区间。
- 外部引用必须解析到 manifest 所在素材批目录内；绝对路径、URL、`..` 逃逸均拒绝。
- 动画、变形目标、相机、灯光、扩展、GLB 等范围外特性直接整批失败，不静默丢数据。
- 缺失图像/骨骼、关节索引越界、权重超出 `[0, 1]` 等问题集中收集、一次性报告，整批不发布。

## 内部模型

对象 ID 由规范化 JSON 的 SHA-256 生成：

- `geo_`：位置、三角形索引、可选法线。
- `tex_`：图像字节、媒体类型（身份只由规范化字节决定）。
- `mat_`：材质参数和 `(texture_id, usage, tex_coord)` 贴图绑定。
- `msh_`：几何引用和 `(material_id, face_offset, face_count)` 材质绑定。
- `skn_`：有序关节节点 ID 与逐关节逆绑定矩阵（4x4 行主序）。
- `ndo_`：4x4 行主序变换、网格引用、可选蒙皮引用、子节点 Merkle 式引用。
- `geo_` 蒙皮版额外包含每顶点 `joint_indices` 与 `weights`（VEC4）。
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

```bash
python3 scripts/create_skinned_sample.py samples/rig
```

第二个样例包含：同一张 `shared.png` 被两种材质作为颜色贴图复用、`packed.png` 分别作为金属粗糙度/法线数据贴图、一个带 `JOINTS_0/WEIGHTS_0` 的两骨骼面板，关节挂在旋转父节点下，可验证转换前后世界空间位置一致。
