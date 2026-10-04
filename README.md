# 三维资源导入服务

第一轮使用受限的 **glTF 2.0 JSON（`.gltf`）** 作为通用交换格式。服务把几何、材质、网格、节点、场景转换为内部内容寻址资源，后续格式只需产出同一套对象和依赖边。

## 支持范围

- glTF 2.0 `.gltf`，内嵌 `data:application/octet-stream;base64` 或同批相对外部 buffer。
- 三角形图元；`POSITION`、可选 `NORMAL`、标量索引；非索引图元会按连续三点转换。
- PBR `pbrMetallicRoughness` 材质因子、`alphaMode`、`doubleSided`、`emissiveFactor`，以及 `baseColorTexture`（颜色用途，sRGB 内容）、`metallicRoughnessTexture`（数据用途，线性内容）、`normalTexture`（含 `scale`）、`occlusionTexture`（含 `strength`）与 `emissiveTexture`。同类材质槽还支持分片版 `baseColorTileTexture`、`metallicRoughnessTileTexture`、`normalTileTexture`、`occlusionTileTexture`、`emissiveTileTexture`。
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
- `ttx_`：分片贴图版本，由用途、`texCoord`、图集宽高和有序分片组成；每片记录位置、实际像素宽高、当前 `img_`，以及更新前来源片和来源图。
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

## 分片贴图选择性版本

顶层 `tileTextures` 用受限 JSON 描述多片图集。分片必须无重叠、无缝覆盖整个图集，且每块声明的宽高必须与图像实际像素一致：

```json
{
  "name": "character-color",
  "usage": "color",
  "atlas_width": 4,
  "atlas_height": 4,
  "tiles": [
    {"id": "head", "x": 0, "y": 0, "width": 2, "height": 2, "source": 0},
    {"id": "body", "x": 2, "y": 0, "width": 2, "height": 2, "source": 1}
  ]
}
```

材质槽使用 `baseColorTileTexture`、`normalTileTexture` 等引用。相同字节仍只有一个 `img_`；即使同一张图出现在不同分片位置，或同时作为颜色/法线/遮蔽解释，也会生成不同的 `ttx_` 分片身份，不会按内容相同而交换坐标或用途。

部分更新仍走 `retarget`，但计划使用 `tile_updates`。未列入 `tiles` 的分片明确按“沿用旧图”处理，绝不会因缺文件而生成空白覆盖：

```json
{
  "tile_updates": [
    {
      "tiled_texture": "character-color",
      "usage": "color",
      "tiles": {"head": {"file": "head-new.png"}},
      "nodes": ["hero"]
    }
  ]
}
```

- 预演报告列出受影响节点、材质、用途、更新片、沿用片、图集尺寸、每片旧/新尺寸、共享边缘和缺失区域；颜色、法线、遮蔽可以分别提供不同完整程度。
- 新分片必须保持原 MIME，宽高为旧片的等比例整数缩小且不放大；位置、用途和共享边缘保持不变，错误会按片聚合，整次发布拒绝。
- `nodes` 只选择对象；被选节点获得新材质/网格/节点版本，未选节点的 `mat_`、`msh_`、`ndo_` ID 完全不变。
- 发布后的 `ttx_` 同时包含新片的 `img_` 来源和沿用片的旧来源；索引 `retargets[].tile_updates` 记录采用对象、每片 updated/retained 状态、边界与来源 ID。重新导出可完整重现组合，后续补齐缺失片或撤回某对象采用时能沿来源链追溯。
- 撤回时使用 `{"withdraw": true, "tiled_texture": "...", "nodes": [...]}`，系统从该版各片的 `source_texture_id/source_tile_id/source_image_id` 恢复来源版本，并只为被选节点生成旧组合材质。
- 预演失败或提交异常时保留既有资产和索引；同一计划重复提交仍得到同一资产 ID，不会发布来源不明的半成品。

### 并行候选与组合采用

同一分片贴图可以在一次 `tile_updates` 中登记多个候选条目，系统并行比较后按对象组合采用，而不是按登记顺序后者覆盖前者：

```json
{
  "tile_updates": [
    {"tiled_texture": "character-color", "tiles": {"head": {"file": "head-a.png"}}, "nodes": ["hero"]},
    {"tiled_texture": "character-color", "priority": 1, "tiles": {"head": {"file": "head-b.png"}}, "nodes": ["hero"]},
    {"tiled_texture": "character-color", "tiles": {"body": {"file": "body-npc.png"}}, "nodes": ["npc"]}
  ]
}
```

- 候选作用于不同对象时各自组合：每个对象按自己命中的候选得到新的 `ttx_` 组合，结果相同的对象共享同一内容寻址身份，未命中任何候选的对象保持旧引用不变。
- 两个候选在同一对象上覆盖同一分片位置且内容不同时构成重叠，必须用明确的 `priority`（整数，大者胜出）解决；缺省或并列时整批拒绝，绝不按最后登记顺序静默覆盖。
- 预演报告中的 `candidates` 逐项列出每个候选的目标对象、每片 `adopted/superseded` 状态与实际采用节点，`base_asset_id` 与 `affected_nodes` 给出原资产与实际影响对象；`changes` 只包含真正被改写的对象。
- 失效候选（文件缺失、尺寸或用途不合法）导致整批拒绝，其文件不会进入对象库；同名同尺寸但内容不同的文件按内容寻址生成不同 `img_`，失效或落选候选不能借用同名同尺寸文件的身份。
- 重复采用同一组合或重复同一撤回都得到同一资产 ID，索引不增加重复资产；撤回某个对象的采用后，其他对象的候选版本保持不变，已发布资产始终可完整导出。

### 分片审阅记录

更新计划与审阅记录分别保存。审阅记录是独立 JSON，保存尚未解决的重叠、已选候选、暂不采用范围，以及每个选择对应的对象、分片贴图、片区和候选文件 SHA-256。

- 使用 `tile-review-save` 写入审阅记录；`--choices` 指向选择列表，`--from-review` 可复制已有记录，`--review` 指定另存目标。
- 保存或另存审阅记录不发布资产；提交时仍使用 `retarget --plan ... --review ... --commit`。
- 重开记录会保留已选、暂不采用和未决重叠；仍有未决重叠或已选候选文件发生变化时，整次发布拒绝。
- 候选文件变化后，只有引用该文件的选择会标记 `needs_reconfirmation: true`，其他选择、暂不采用范围和未决项不会被清空。
- 另存记录可比较不同选择；切换记录本身不会改变库中资产，局部采用、撤回校验、失效候选整批拒绝和重复操作去重继续有效。

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
