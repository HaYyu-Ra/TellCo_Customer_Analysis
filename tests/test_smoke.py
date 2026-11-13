from src.tellco.analytics import analyze
import pandas as pd

def test_pipeline_smoke():
    rows=[]
    for user in range(1,7):
        for session in range(3):
            rows.append({'msisdn_number':user,'duration_s':20+user+session,'total_dl_bytes':1000*user,'total_ul_bytes':100*user,'tcp_dl_retrans_vol_bytes':user*3,'tcp_ul_retrans_vol_bytes':user,'avg_rtt_dl_ms':10+user,'avg_rtt_ul_ms':12+user,'avg_bearer_tp_dl_kbps':100*user,'avg_bearer_tp_ul_kbps':50*user,'youtube_dl_bytes':500*user,'youtube_ul_bytes':10*user,'handset_type':'Device X'})
    result=analyze(pd.DataFrame(rows))
    assert len(result['customers'])==6
    assert result['scored']['satisfaction_score'].notna().all()
    assert len(result['apps'])==7
