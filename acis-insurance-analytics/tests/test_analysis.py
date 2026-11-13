import pandas as pd
from acis.analysis import aggregate_policies

def test_policy_aggregation():
    rows=pd.DataFrame({"PolicyID":[1,1,2],"Province":["Gauteng","Gauteng","Western Cape"],"PostalCode":[1000,1000,8000],"Gender":["Male","Male","Female"],"TotalPremium":[100.,50.,80.],"TotalClaims":[0.,20.,0.],"VehicleType":["Car","Car","Car"],"RegistrationYear":[2010,2010,2015],"make":["A","A","B"],"Model":["X","X","Y"]})
    p=aggregate_policies(rows).set_index("PolicyID")
    assert len(p)==2 and p.loc[1,"premium"]==150 and p.loc[1,"claims"]==20 and p.loc[1,"has_claim"]==1
