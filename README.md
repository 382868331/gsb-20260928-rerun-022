# 三维资源导入服务

第一轮使用受限的 **glTF 2.0 JSON（`.gltf`）** 作为通用交换格式。服务把几何、材质、网格、节点、场景转换为内部内容寻址资源，后续格式只需产出同一套对象和依赖边。

## 支持范围

- glTF 2.0 `.gltf`，内嵌 `data:application/octet-stream;base64` 或同批相对外部 buffer。
- 三角形图元；`POSITION`、可选 `NORMAL`、标量索引；非索引图元会按连续三点转换。
- PBR `pbrMetallicRoughness` 材质因子、`alphaMode`、`doubleSided`、`emissiveFactor`，以及 `baseColorTexture`（颜色用途，sRGB 内容）、`metallicRoughnessTexture`（数据用途，线性内容）、`normalTexture`（含 `scale`）、`occlusionTexture`（含 `strength`）与 `emissiveTexture`；贴图支持单图或按用途组织的多张分片（见“分片贴图的选择性版本替换”）。
- 贴图来源支持外部相对文件、内嵌 `data:` 与 `bufferView`（PNG/JPEG）。同一图像被多个材质或多种用途引用时共享同一 `img_` 身份；颜色、数据、法线、遮蔽、自发光用途分别生成独立的 `tex_` 身份，数据贴图不会被当成颜色处理。
- 蒙皮网格：`JOINTS_0`/`WEIGHTS_0`（VEC4，整数关节、浮点权重）、`skins[].joints`、可选 `skeleton` 与列主序 `inverseBindMatrices`（MAT4）。
- 节点 TRS 或 16 元素矩阵、嵌套层级和多图元材质区间；蒙皮节点通过 `skin` 引用皮肤对象。
- 外部引用必须解析到 manifest 所在素材批目录内；绝对路径、URL、`..` 逃逸均拒绝。
- 动画、变形目标、相机、灯光、扩展、GLB 等范围外特性直接整批失败，不静默丢数据。

## 内部模型

对象 ID 由规范化 JSON 的 SHA-256 生成：

- `geo_`：位置、三角形索引、可选法线、可选每顶点 4 关节索引与 4 权重。
- `img_`：图像 MIME 与字节内容（内容寻址，天然去重）。
- `tex_`：`image_id` + 用途（`color`/`data`/`normal`/`occlusion`/`emissive`）+ `texCoord`。
- `mat_`：材质参数、自发光因子、法线缩放、遮蔽强度与各用途贴图引用。
- `msh_`：几何引用和 `(material_id, face_offset, face_count)` 材质绑定。
- `skn_`：关节节点引用列表与每关节 4x4 逆绑定矩阵。
- `ndo_`：4x4 行主序变换、网格引用、子节点 Merkle 式引用、可选 `skin_id`。
- `scn_`：根节点集合。
- `ast_`：一次导入产出的资源根。

相同内容复用同一对象文件，因此相同几何/贴图在网格与材质间共享身份。导入先完成解析和完整错误收集，再写入对象与索引；缺失图像/骨骼、关节或权重越界、损坏或非法依赖时不生成索引，不发布半成品。对象文件和 `index.json` 都通过临时文件原子替换，再次导入只会复用既有对象，旧版本资产的闭包保持不变。

蒙皮求值可由 `resource3d.math3d` 完成：`skin_point(point, joints, weights, joint_world, inverse_bind_matrices)` 按权重混合各关节的 `世界变换 × 逆绑定矩阵`；`inverse()` 提供矩阵求逆。

## 材质更新对照

对同一素材局部修改材质后重新导入，可用 `diff` 得到资源对照：它按材质名匹配新旧版本，列出每种贴图用途的新增、移除和变化（含新旧 `tex_` 与其 `img_` 引用），并汇总使用该材质的网格与节点，以及新旧资产闭包中增删的贴图与图像。未变化的共享材质/贴图/图像保持同一内容寻址身份，旧版本导出不受影响。

```bash
python3 -m resource3d.cli diff \
  --library build/library \
  --old-asset "$OLD_ASSET" \
  --new-asset "$NEW_ASSET" \
  --output build/material-diff.json
```

## 贴图分辨率替换

导入后可为共享贴图切换分辨率版本（例如为部分物体选用低分辨率图），材质参数、用途与外观解释保持不变。替换计划是一个 JSON 文件，每项指定源图像（`img_` ID 或图像名）、用途、替换文件，以及可选的节点名单（缺省表示所有引用该贴图的物体）：

```json
{
  "replacements": [
    {"image": "shared", "usage": "color", "file": "shared-low.png", "nodes": ["node-a"]}
  ]
}
```

```bash
# 预演：列出将切换版本的材质与物体，或报告拒绝原因，不写入任何资源
python3 -m resource3d.cli retarget --library build/library --asset "$ASSET" --plan plan.json
# 提交：全部校验通过才原子发布新资产版本，失败不留下半套资源
python3 -m resource3d.cli retarget --library build/library --asset "$ASSET" --plan plan.json --commit
# 回退：返回生成该版本的原资产 ID（原版本闭包始终完整可导出）
python3 -m resource3d.cli rollback --library build/library --asset "$NEW_ASSET"
```

规则与行为：

- 替换图像必须存在、与原图 MIME 一致，且宽高是原图的整数倍缩小（等比例、不放大），否则按“图像缺失 / 尺寸关系不合法 / 用途不兼容”逐项报告并整体拒绝。
- 同一图像同时承担颜色与数据等用途时，各用途的替换记录相互独立：只换颜色用途不会影响法线、遮蔽或数据贴图。
- 只选择部分物体时，被选节点得到新的材质/网格/节点版本，未选对象保持原有内容寻址身份，共享关系不会把它们一起带走。
- 新图像与贴图仍是内容寻址对象：重复执行同一计划得到同一资产 ID，索引中的 `retargets` 记录只保留一份来源（基资产 + 各项替换），不会产生内容相同但来源不清的资源。
- 重新导出新资产即可复现各物体的分辨率选择；`rollback` 结合原资产 ID 可回到原图版本。

## 分片贴图的选择性版本替换

同一种共享材质的每个贴图用途可以由多张**分片**组成（例如 2x2 拼接的颜色贴图）。分片身份是 `图像 + 用途 + 分片位置 + 布局 + 边缘标签`：

- `img_` 仍只按字节内容去重；同一张图被不同位置或颜色/法线等不同用途引用时只导入一次。
- 每个位置、每个用途生成各自独立的 `tex_` 身份。内容相同也不会交换位置或用途解释：颜色片不会被当作法线片。
- `Material` 的 `tile_textures` 按用途保存有序的分片槽；单图用途仍使用原来的单张贴图字段。

分片布局通过 glTF 的 `extras` 描述：`textures[i].extras.tile` 给出 `position`（列、行）、`layout`（列数、行数）与可选 `edges`（左/右/上/下四组接缝标签），`materials[i].extras.tiles` 按用途列出分片（`index`、可选 `texCoord/extras`）。导入时校验位置不重复、覆盖完整布局、`texCoord` 一致，以及相邻分片接缝处的边缘标签一致；缺片的导入直接失败，不会用空白图补齐。

分片版本更新计划与分辨率替换共用 `retarget` 命令，计划顶层含 `tiles` 时走分片流程：

```json
{
  "tiles": [
    {"material": "tiled-mat", "usage": "color", "position": [0, 0], "file": "color-0-0-low.png"},
    {"material": "tiled-mat", "usage": "normal", "position": [1, 0], "file": "normal-1-0-low.png",
     "edges": [[], ["seam"], [], []]}
  ],
  "nodes": ["node-a"],
  "exclude_nodes": []
}
```

- 一次更新可以只给部分分片；颜色、法线、遮蔽等用途的完整程度可以不同。未列出的分片在报告的 `kept_tiles` 中显式列出并继续沿用旧来源，绝不会被解释成空白覆盖。
- `nodes` 缺省表示所有受影响物体；只列部分节点时，仅被选对象生成新的材质/网格/节点版本，未选对象保持原内容寻址身份，共享材质不会把它们一并切换。`exclude_nodes` 可显式标记暂不接受更新的区域。
- 预演报告 `updates` 给出每项的用途、位置、旧/新分辨率、MIME、自身边缘标签与邻片接缝标签，以及每个材质变体的新 `img_`/`tex_` 身份；`regions` 列出受影响的物体、网格、材质和面区间（`face_offset/face_count`）。先预演确认跨分片边界外观与缺失区域，再 `--commit`。
- 校验包括：材质/用途/分片位置存在、文件可读、MIME 不变、新尺寸是旧尺寸的等比例整数倍缩小、同一 `(材质, 用途)` 批次比例一致、`edges`（如提供）与原片一致、节点可解析。任一失败都整体拒绝，保留已有材质与未发布选择，不写入半成品资源。
- 发布资产的 `tile_versions` 记录形成可完整重现的材质版本：根资产、采用节点、每个更新分片在各材质变体上的旧 `tex_/img_` 来源、继续沿用的分片清单。重复同一计划得到同一资产 ID；同内容文件只产生一个 `img_`，但每个位置/用途保留各自采用身份。
- 撤回某对象的采用使用 `{"revert": ["node-a"]}` 计划：被列节点回到根资产的材质组合，其他节点保持现状；溯源链始终指向最初导入的根资产，重新导出再导入即可追到当时各片来源。

```bash
python3 scripts/create_tiled_sample.py samples/tiled
python3 -m resource3d.cli retarget --library build/library --asset "$ASSET" --plan samples/tiled/tile-plan.json
python3 -m resource3d.cli retarget --library build/library --asset "$ASSET" --plan samples/tiled/tile-plan.json --commit
```

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

`samples/character` 样例包含：被两种材质共享的颜色贴图、线性金属粗糙度数据贴图、外部 `.bin` 与内嵌 buffer 中的几何/权重/逆绑定矩阵，以及嵌套关节层级（父节点旋转 90°）。导出的 JSON 可直接检查所有面、贴图用途、关节引用与完整层级变换。
