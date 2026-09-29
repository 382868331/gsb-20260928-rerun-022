# r3d — 三维资源导入服务

格式中立的三维资源模型与导入管线。第一轮只支持一个明确限定范围的 glTF 2.0 子集；后续格式直接产出同一套资源类型、内容身份与依赖关系，不需要修改资源库。

## 内部资源模型

资源分五类，全部为不可变内容寻址：

| 类型 | 内容 | 身份 |
| --- | --- | --- |
| `geometry` | 位置、法线、UV、三角形索引 | `geometry-` + 规范化 JSON 的 SHA-256 |
| `material` | PBR baseColor 因子、金属度/粗糙度、贴图引用 | `material-` + 哈希 |
| `image` | 媒体类型、字节数与 blob 引用 | `image-` + 哈希 |
| `node` | 列主序 4x4 局部矩阵、子节点、几何/材质绑定 | `node-` + 哈希 |
| `scene` | 根节点列表 | `scene-` + 哈希 |

图片字节保存在 `blobs/blob-<sha256>.bin`。身份只由内容决定，所以同一资源跨文件、跨批次、跨格式重复导入都复用同一条记录。引用只能指向带类型前缀的资源 ID，模型层会拒绝错配类型。

## 支持的 glTF 范围

- `.gltf` 与 `.glb`；外部 buffer/图片、base64 data URI、GLB BIN chunk。
- `TRIANGLES` 图元；`POSITION`（必需）、`NORMAL`、`TEXCOORD_0`；支持索引/无索引几何。
- accessor：SCALAR/VEC2/VEC3/VEC4，BYTE..FLOAT；索引限 UNSIGNED_BYTE/SHORT/INT。
- PBR metallic-roughness：baseColor 因子与 baseColorTexture。
- 节点层级：`matrix` 或 TRS（四元数必须归一化）。

明确拒绝：skin、morph target、稀疏 accessor、交错顶点 bufferView、非三角形模式、相机/灯光/动画、任何 `extensionsUsed/extensionsRequired`、网络 URI、绝对路径以及任何解析出批次目录的相对路径。

## 批次语义

1. 递归收集批次目录中的 `.gltf/.glb`，逐文件累积全部问题。
2. 仅当整批没有任何问题时才进入暂存与发布；损坏文件、缺失/越界依赖会整批拒绝。
3. 发布通过同目录临时文件加 `os.replace` 原子落盘，失败不留半成品。
4. 已存在的内容自动复用，报告分别给出新增与复用计数。

## 使用

```bash
# 生成自带样例（共享几何 + 嵌套节点 + 贴图）和损坏批次
python3 scripts/create_sample.py --output samples
python3 -m r3d import --library ./library --batch samples/sample_batch
python3 -m r3d list --library ./library
python3 -m r3d verify --library ./library <scene-id>
python3 -m r3d export --library ./library --scene <scene-id> --output out/export.json
```

导出的 JSON 是完整闭包：包含每个资源、前序节点顺序、每个节点的世界矩阵、每个几何实例的世界空间采样点，以及几何复用统计，可直接核对变换、三角形数量与材质/图片引用。

## 测试

```bash
python3 -m unittest discover -s tests
```

## 后续接入新格式

新增格式只需解析为 `r3d.model` 的构造函数输出（`make_geometry`、`make_material`、`make_image`、`make_node`、`make_scene`），再交给 `r3d.store.Library` 提交。不要在内部模型中保留来源格式概念；格式特有语义应在导入器中转换或显式报错。
