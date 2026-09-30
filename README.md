# 三维资源导入服务

第一轮使用受限的 **glTF 2.0 JSON（`.gltf`）** 作为通用交换格式。服务把几何、材质、网格、节点、场景转换为内部内容寻址资源，后续格式只需产出同一套对象和依赖边。

## 支持范围

- glTF 2.0 `.gltf`，内嵌 `data:application/octet-stream;base64` 或同批相对外部 buffer。
- 三角形图元；`POSITION`、可选 `NORMAL`、标量索引；非索引图元会按连续三点转换。
- PBR `pbrMetallicRoughness` 材质因子、`emissiveFactor`、`alphaMode`、`doubleSided`，以及 `baseColorTexture`（颜色用途，sRGB 内容）、`metallicRoughnessTexture`（数据用途，线性内容）、`normalTexture`（法线用途，切线空间数据）、`occlusionTexture`（遮蔽用途）与 `emissiveTexture`（自发光用途，sRGB 内容）；`normalTexture.scale`、`occlusionTexture.strength` 等仅颜色/数据之外的修饰参数不参与身份，但不会被当作颜色贴图处理。
- 贴图来源支持外部相对文件、内嵌 `data:` 与 `bufferView`（PNG/JPEG）。同一图像被多个材质或多种用途引用时共享同一 `img_` 身份；颜色/数据/法线/遮蔽/自发光用途分别生成独立的 `tex_` 身份，不会互相混淆。
- 蒙皮网格：`JOINTS_0`/`WEIGHTS_0`（VEC4，整数关节、浮点权重）、`skins[].joints`、可选 `skeleton` 与列主序 `inverseBindMatrices`（MAT4）。
- 节点 TRS 或 16 元素矩阵、嵌套层级和多图元材质区间；蒙皮节点通过 `skin` 引用皮肤对象。
- 外部引用必须解析到 manifest 所在素材批目录内；绝对路径、URL、`..` 逃逸均拒绝。
- 动画、变形目标、相机、灯光、扩展、GLB 等范围外特性直接整批失败，不静默丢数据。

## 内部模型

对象 ID 由规范化 JSON 的 SHA-256 生成：

- `geo_`：位置、三角形索引、可选法线、可选每顶点 4 关节索引与 4 权重。
- `img_`：图像 MIME 与字节内容（内容寻址，天然去重）。
- `tex_`：`image_id` + 用途（`color`/`data`/`normal`/`occlusion`/`emissive`）+ `texCoord`。
- `mat_`：材质参数与颜色/数据/法线/遮蔽/自发光贴图引用。
- `msh_`：几何引用和 `(material_id, face_offset, face_count)` 材质绑定。
- `skn_`：关节节点引用列表与每关节 4x4 逆绑定矩阵。
- `ndo_`：4x4 行主序变换、网格引用、子节点 Merkle 式引用、可选 `skin_id`。
- `scn_`：根节点集合。
- `ast_`：一次导入产出的资源根。

相同内容复用同一对象文件，因此相同几何/贴图在网格与材质间共享身份。导入先完成解析和完整错误收集，再写入对象与索引；缺失图像/骨骼、关节或权重越界、损坏或非法依赖时不生成索引，不发布半成品。对象文件和 `index.json` 都通过临时文件原子替换，再次导入只会复用既有对象，旧版本资产的闭包保持不变。

## 材质局部更新对照

两个已发布资产版本之间可生成材质级资源对照：

```bash
python3 -m resource3d.cli diff-materials \
  --library build/library --old OLD_ASSET_ID --new NEW_ASSET_ID \
  --output build/material-diff.json
```

或调用 `ResourceLibrary.material_update_report(old_asset_id, new_asset_id)`。报告按材质名匹配，仅列出内容变化的材质，包含：

- `usage_changes`：每个发生变化的贴图用途（`added`/`removed`/`replaced`）、变化前后的 `tex_` 与其底层 `img_` 引用；同一图像换用途时图像引用保持不变。
- `factor_changes`：材质因子（含 `emissive_factor`）与渲染状态的前后值。
- `users_before`/`users_after`：使用该材质的网格与节点对象名。
- `added_materials`/`removed_materials`：重命名或新增/删除的材质名。

更新材质会产生新的 `mat_` 身份，只波及实际引用它的网格/节点闭包；共享同一材质或同一颜色贴图的其他对象在重新导出时保持逐字节一致。

蒙皮求值可由 `resource3d.math3d` 完成：`skin_point(point, joints, weights, joint_world, inverse_bind_matrices)` 按权重混合各关节的 `世界变换 × 逆绑定矩阵`；`inverse()` 提供矩阵求逆。

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

`samples/robot` 样例包含：两个实例共享的立方体几何、一个外部 `.bin` 金字塔、两种材质、父节点旋转以及嵌套平移/缩放。

```bash
python3 scripts/create_skinned_sample.py samples/character
python3 -m resource3d.cli import --library build/library --manifest samples/character/character.gltf
```

`samples/character` 样例包含：被两种材质共享的颜色贴图、线性金属粗糙度数据贴图、共享同一图像但分别解释为法线与遮蔽的两张贴图、自发光贴图与 `emissiveFactor`、外部 `.bin` 与内嵌 buffer 中的几何/权重/逆绑定矩阵，以及嵌套关节层级（父节点旋转 90°）。导出的 JSON 可直接检查所有面、贴图用途、关节引用与完整层级变换。
