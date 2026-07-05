from pathlib import Path
import pandas as pd
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = Path("./ragtest")
OUT = ROOT / "output"
VIS = OUT / "visuals"
VIS.mkdir(parents=True, exist_ok=True)

# 한글 폰트 설정: Windows 기준 맑은 고딕
font_path = Path("C:/Windows/Fonts/malgun.ttf")

if font_path.exists():
    font_name = fm.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["font.family"] = font_name
else:
    plt.rcParams["font.family"] = "Malgun Gothic"

plt.rcParams["axes.unicode_minus"] = False

TABLES = [
    "documents",
    "text_units",
    "entities",
    "relationships",
    "communities",
    "community_reports",
    "covariates",
]

def read_parquet_table(name: str) -> pd.DataFrame:
    path = OUT / f"{name}.parquet"
    if not path.exists():
        print(f"[MISSING] {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    print(f"[OK] {name}: rows={len(df)}, cols={len(df.columns)}")
    print(f"     columns={list(df.columns)}")
    return df

print("=== GraphRAG Output Inspection ===")

dfs = {}
for table in TABLES:
    dfs[table] = read_parquet_table(table)

print("\n=== Save preview CSV files ===")

for name, df in dfs.items():
    if not df.empty:
        preview_path = VIS / f"{name}_preview.csv"
        df.head(50).to_csv(preview_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {preview_path}")

entities = dfs.get("entities", pd.DataFrame())
relationships = dfs.get("relationships", pd.DataFrame())
communities = dfs.get("communities", pd.DataFrame())
community_reports = dfs.get("community_reports", pd.DataFrame())

# 1. Top entities
if not entities.empty:
    sort_cols = []
    for col in ["degree", "frequency"]:
        if col in entities.columns:
            sort_cols.append(col)

    if sort_cols:
        top_entities = entities.sort_values(sort_cols, ascending=False).head(30)
    else:
        top_entities = entities.head(30)

    top_entities.to_csv(VIS / "top_entities.csv", index=False, encoding="utf-8-sig")
    print("\n[SAVE] top_entities.csv")

    if "title" in top_entities.columns and "degree" in top_entities.columns:
        plt.figure(figsize=(10, 8))
        top_entities_plot = top_entities[["title", "degree"]].dropna().head(20)
        top_entities_plot = top_entities_plot.sort_values("degree")
        plt.barh(top_entities_plot["title"], top_entities_plot["degree"])
        plt.title("Top Entities by Degree")
        plt.xlabel("Degree")
        plt.tight_layout()
        plt.savefig(VIS / "top_entities_by_degree.png", dpi=200)
        plt.close()
        print("[SAVE] top_entities_by_degree.png")

# 2. Top relationships
if not relationships.empty:
    if "weight" in relationships.columns:
        top_relationships = relationships.sort_values("weight", ascending=False).head(50)
    else:
        top_relationships = relationships.head(50)

    top_relationships.to_csv(VIS / "top_relationships.csv", index=False, encoding="utf-8-sig")
    print("[SAVE] top_relationships.csv")

    if "weight" in relationships.columns:
        plt.figure(figsize=(10, 6))
        relationships["weight"].dropna().plot(kind="hist", bins=30)
        plt.title("Relationship Weight Distribution")
        plt.xlabel("Weight")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(VIS / "relationship_weight_distribution.png", dpi=200)
        plt.close()
        print("[SAVE] relationship_weight_distribution.png")

# 3. Community reports
if not community_reports.empty:
    cols = [c for c in ["community", "level", "title", "summary", "rank", "full_content"] if c in community_reports.columns]
    community_reports[cols].head(30).to_csv(
        VIS / "community_reports_preview.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print("[SAVE] community_reports_preview.csv")

# 4. Community size plot
if not communities.empty and "size" in communities.columns:
    plt.figure(figsize=(10, 6))
    communities["size"].dropna().plot(kind="hist", bins=30)
    plt.title("Community Size Distribution")
    plt.xlabel("Community Size")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(VIS / "community_size_distribution.png", dpi=200)
    plt.close()
    print("[SAVE] community_size_distribution.png")

# 5. Interactive graph visualization
if not entities.empty and not relationships.empty:
    print("\n=== Building interactive graph HTML ===")

    top_n = 150

    if "degree" in entities.columns:
        selected_entities = set(
            entities.sort_values("degree", ascending=False)
            .head(top_n)["title"]
            .astype(str)
        )
    else:
        selected_entities = set(entities.head(top_n)["title"].astype(str))

    rel = relationships.copy()

    required_cols = {"source", "target"}
    if required_cols.issubset(set(rel.columns)):
        rel = rel[
            rel["source"].astype(str).isin(selected_entities)
            & rel["target"].astype(str).isin(selected_entities)
        ]

        if "weight" in rel.columns:
            rel = rel.sort_values("weight", ascending=False).head(300)
        else:
            rel = rel.head(300)

        G = nx.Graph()

        entity_info = entities.set_index(entities["title"].astype(str)).to_dict("index")

        for title in selected_entities:
            info = entity_info.get(title, {})
            degree = info.get("degree", 1)
            entity_type = info.get("type", "")
            description = str(info.get("description", ""))[:500]

            G.add_node(
                title,
                label=title,
                title=f"type: {entity_type}<br>degree: {degree}<br>{description}",
                size=max(5, min(50, int(degree) if pd.notna(degree) else 5)),
            )

        for _, row in rel.iterrows():
            source = str(row["source"])
            target = str(row["target"])
            weight = row.get("weight", 1)
            description = str(row.get("description", ""))[:500]

            if source in G.nodes and target in G.nodes:
                G.add_edge(
                    source,
                    target,
                    value=float(weight) if pd.notna(weight) else 1.0,
                    title=description,
                )

        net = Network(height="850px", width="100%", notebook=False, cdn_resources="in_line")
        net.from_nx(G)

        html_path = VIS / "knowledge_graph_top_entities.html"

        # Windows cp949 인코딩 오류 방지를 위해 UTF-8로 직접 저장
        html = net.generate_html()
        html_path.write_text(html, encoding="utf-8")

        print(f"[SAVE] {html_path}")
    else:
        print("[SKIP] relationships.parquet에 source/target 컬럼이 없어 graph HTML 생성 생략.")

print("\n=== Done ===")
print(f"결과 폴더: {VIS.resolve()}")