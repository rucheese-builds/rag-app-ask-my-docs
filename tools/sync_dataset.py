import pandas as pd
from pathlib import Path
import sys

# Define paths
WORKSPACE_CSV = Path("evaluation/golden_dataset.csv")
WORKSPACE_XLSX = Path("evaluation/golden_dataset.xlsx")
ARTIFACT_XLSX = Path("/Users/ruchiagarwal/.gemini/antigravity/brain/b7bfe303-91f3-4dad-a6fe-b3b4f76d0954/golden_dataset.xlsx")

def sync():
    print("=== Synchronizing Golden Dataset Excel to CSV ===")
    
    # 1. Determine which Excel file to prioritize (workspace vs. artifacts directory)
    if WORKSPACE_XLSX.exists():
        excel_source = WORKSPACE_XLSX
        print(f"[Sync] Found Excel sheet in workspace: {WORKSPACE_XLSX}")
    elif ARTIFACT_XLSX.exists():
        excel_source = ARTIFACT_XLSX
        print(f"[Sync] Found Excel sheet in artifacts directory: {ARTIFACT_XLSX}")
    else:
        print("[Sync] ERROR: No golden_dataset.xlsx file found to sync from!")
        sys.exit(1)
        
    try:
        # 2. Read the Excel sheet
        df = pd.read_excel(excel_source)
        
        # Ensure mandatory columns exist
        required_cols = ["question", "ground_truth", "mandatory_keywords", "category", "relevant_sources", "parent_chunk_id", "page_number"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = "" # pad missing columns
                
        # Fill NaN values with empty string
        df = df.fillna("")
        
        # 3. Write back to workspace CSV
        df.to_csv(WORKSPACE_CSV, index=False, encoding="utf-8")
        print(f"[Sync] Successfully updated CSV: {WORKSPACE_CSV} ({len(df)} questions)")
        
        # 4. Keep Excel files in sync in both locations
        if excel_source == WORKSPACE_XLSX:
            df.to_excel(ARTIFACT_XLSX, index=False)
            print(f"[Sync] Copied updated Excel sheet to artifacts panel: {ARTIFACT_XLSX}")
        else:
            df.to_excel(WORKSPACE_XLSX, index=False)
            print(f"[Sync] Copied updated Excel sheet to workspace: {WORKSPACE_XLSX}")
            
    except Exception as e:
        print(f"[Sync] ERROR during synchronization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sync()
