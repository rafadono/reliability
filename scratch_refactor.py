import sys

def main():
    with open('backend/api/analysis.py', 'r', encoding='utf-8') as f:
        content = f.read()

    imports_to_add = '''
import time
from src.reliability_analysis.utils.config import NLP_MODELS_TO_COMPARE
from src.reliability_analysis.analysis.hf_classifier import SemanticModelManager
'''

    content = content.replace('import traceback\n', 'import traceback\n' + imports_to_add)

    new_endpoint = '''@router.post("/analysis/comment-mining", tags=["Analysis"])
async def comment_mining(req: AnalysisRequest) -> Dict[str, Any]:
    if state.filter_manager is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        data = state.filter_manager.get_filtered_data()
        if data.empty:
            return {"status": "success", "results": {}}

        comment_col = None
        for col in data.columns:
            if col.lower() in ("comentario", "comment"):
                comment_col = col
                break

        if not comment_col:
            return {"status": "warning", "message": "No comment column found in dataset", "results": {}}

        comments_df = data.dropna(subset=[comment_col])
        valid_mask = ~comments_df[comment_col].astype(str).str.lower().isin(["---", "nan", "none", "null", "no aplica", "n/a"])
        valid_data = comments_df[valid_mask]

        total_records = len(data)
        comments_count = len(valid_data)
        coverage = (comments_count / total_records * 100.0) if total_records > 0 else 0.0

        if valid_data.empty:
            return {"status": "success", "results": {}, "coverage": 0.0, "total_comments": 0}

        core_stop_words = {"de","la","el","y","en","por","que","para","un","una","con","se","no","del","al","lo","los","las","es","su","sus","como","más","mas","o","pero","este","esta","ha","debido","realiza","presenta","genera","causa","además","ademas","sobre","entre","otro","otra","otros","otras","caso","hace","hecho","estos","estas","sino","toda","todo","todos","todas","donde","desde","hasta","cuando","quien","cual","cuales","muy","sólo","solo","cada","bien","también","tambien","tampoco","después","despues","antes","ahora","the","of","and","to","in","is","you","that","it","he","was","for","on","are","as","with","his","they","i","at","be","this","have","from","or","one","had","by","but","not","what","all","were","we","when","your","can","said","there","use","an","each","which","she","do","how","their","if","will","up","other","about","out","many","then"}
        generic_words = {"decision","decisión","operacional","operacion","operación","detencion","detención","detenida","detenido","detencin","proceso","procesos","sistema","sistemas","falla","fallas","parada","paradas","bajo","alto","nivel","niveles","vacio","vacío","corriendo","tiempo","tiempos","minuto","minutos","hora","horas","días","día","sección","área","operario","operarios","turno","turnos","reporte","reportes","observación","observaciones","trabajo","trabajos","personal","inspección","inspeccion","revisión","revision","registro","registros","código","codigo","estado","estados","actividad","actividades","inicio","fin","espera","esperas","valor","valores","equipo","equipos","problema","problemas","evento","eventos","motivo","motivos","failure","failures","stop","stops","stopped","system","systems","downtime","uptime","running","code","codes","activity","activities","waiting","process","operational","operation","operations","area","shift","shifts","operator","report","reports","observation","observations","inspection","inspections"}

        def clean_word(w: str) -> str:
            w = w.lower()
            glitches = {"detencin":"detencion","proteccin":"proteccion","calibracin":"calibracion","operacin":"operacion","decisin":"decision","observacin":"observacion","inspeccin":"inspeccion","revisin":"revision","obstruccin":"obstruccion","ventilacin":"ventilacion","alimentacin":"alimentacion"}
            return glitches.get(w, w)

        def reduce_plural(w: str) -> str:
            if w.endswith("ces"): return w[:-3] + "z"
            if w.endswith("es") and len(w) > 4:
                c = w[:-2]
                if c[-1] in "bcdfghjklmnpqrstvwxyz": return c
            if w.endswith("s") and len(w) > 3 and not w.endswith("is") and not w.endswith("us"):
                return w[:-1]
            return w

        from collections import Counter
        import re

        all_terms = []
        texts_list = []
        
        for _, row in valid_data.iterrows():
            text = str(row[comment_col])
            texts_list.append(text)
            text_lower = text.lower()
            words = re.findall(r"\\b[a-zA-Záéíóúñ]{3,}\\b", text_lower)
            cleaned_words = [reduce_plural(clean_word(w)) for w in words]
            for w in cleaned_words:
                if w not in core_stop_words and w not in generic_words and len(w) > 2:
                    all_terms.append(w)
            for i in range(len(cleaned_words) - 1):
                w1, w2 = cleaned_words[i], cleaned_words[i + 1]
                if w1 not in core_stop_words and w2 not in core_stop_words and (w1 not in generic_words or w2 not in generic_words):
                    all_terms.append(f"{w1} {w2}")

        counter = Counter(all_terms)
        top_keywords = [{"word": word, "count": count} for word, count in counter.most_common(15)]

        type_col = "Type" if "Type" in data.columns else None
        mode_col = "mdf"
        
        final_results = {}
        target_labels = ["Operational", "Cleaning/Blockage", "Mechanical", "Electrical", "Instrumentation/Failure", "Others"]

        for model_name in NLP_MODELS_TO_COMPARE:
            start_time = time.time()
            analyzed_records = []
            
            if model_name == "Legacy Keyword NLP":
                for _, row in valid_data.iterrows():
                    text_lower = str(row[comment_col]).lower()
                    cat = "Others"
                    if any(k in text_lower for k in ["operacional", "operación", "decision", "decisión", "operational", "operation", "process", "operator"]): cat = "Operational"
                    elif any(k in text_lower for k in ["limpieza", "atollo", "obstrucción", "obstruido", "cleaning", "blockage", "jam", "clog", "obstructed"]): cat = "Cleaning/Blockage"
                    elif any(k in text_lower for k in ["mecánico", "mecanico", "perno", "shaft", "eje", "rodamiento", "bearing", "correa", "motor", "mechanical", "bolt", "belt"]): cat = "Mechanical"
                    elif any(k in text_lower for k in ["eléctrico", "electrico", "cable", "bobina", "fase", "breaker", "contacto", "potencia", "electrical", "coil", "phase", "contact", "power"]): cat = "Electrical"
                    elif any(k in text_lower for k in ["falla", "alarma", "sensor", "calibracion", "calibración", "instrumentación", "instrumento", "failure", "alarm", "calibration", "instrumentation", "instrument"]): cat = "Instrumentation/Failure"
                    
                    analyzed_records.append({"category": cat, "type": str(row.get(type_col, "Unknown")), "mode": str(row.get(mode_col, "Unknown"))})
            else:
                candidate_labels = ["Operational", "Cleaning or Blockage", "Mechanical", "Electrical", "Instrumentation or Failure", "Others"]
                try:
                    predictions = SemanticModelManager.batch_predict(texts_list, model_name, candidate_labels)
                    label_map = {"Cleaning or Blockage": "Cleaning/Blockage", "Instrumentation or Failure": "Instrumentation/Failure"}
                    for i, (_, row) in enumerate(valid_data.iterrows()):
                        raw_cat = predictions[i] if i < len(predictions) else "Others"
                        cat = label_map.get(raw_cat, raw_cat)
                        if cat not in target_labels: cat = "Others"
                        analyzed_records.append({"category": cat, "type": str(row.get(type_col, "Unknown")), "mode": str(row.get(mode_col, "Unknown"))})
                except Exception as ex:
                    logger.error(f"Error running model {model_name}: {ex}")
                    for _, row in valid_data.iterrows():
                        analyzed_records.append({"category": "Others", "type": str(row.get(type_col, "Unknown")), "mode": str(row.get(mode_col, "Unknown"))})

            categories_details = []
            for cat_name in target_labels:
                cat_records = [r for r in analyzed_records if r["category"] == cat_name]
                count = len(cat_records)
                top_types = [t for t, _ in Counter([r["type"] for r in cat_records]).most_common(3)]
                top_modes = [m for m, _ in Counter([r["mode"] for r in cat_records]).most_common(3)]
                categories_details.append({"category": cat_name, "count": count, "top_types": top_types, "top_modes": top_modes})
                
            execution_time = round(time.time() - start_time, 2)
            final_results[model_name] = {
                "categories": categories_details,
                "keywords": top_keywords,
                "execution_time_seconds": execution_time
            }

        return {
            "status": "success",
            "coverage": float(coverage),
            "total_comments": int(comments_count),
            "results": final_results
        }
    except Exception as e:
        logger.error(f"Comment mining error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
'''

    parts = content.split('@router.post("/analysis/comment-mining", tags=["Analysis"])')
    new_content = parts[0] + new_endpoint

    with open('backend/api/analysis.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Successfully refactored analysis.py")

if __name__ == "__main__":
    main()
