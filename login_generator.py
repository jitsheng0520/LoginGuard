import numpy as np
import pandas as pd

n=1000

# -------------------
# Normal logins
# -------------------
data = {
"login_hour":np.random.normal(14,4,n).astype(int),
"failed_attempts":np.random.poisson(1,n),
"new_device":np.random.binomial(1,0.05,n),
"ip_changed":np.random.binomial(1,0.08,n),
"geo_distance_km":np.random.normal(20,10,n),
"session_duration":np.random.normal(30,8,n),
"label":["normal"]*n
}

df=pd.DataFrame(data)

# clean impossible negatives
df["login_hour"]=df["login_hour"].clip(0,23)
df["geo_distance_km"]=df["geo_distance_km"].clip(0)

# -------------------
# Inject anomalies
# -------------------
num_anomalies=50

anomalies=pd.DataFrame({
"login_hour":np.random.randint(1,4,num_anomalies),      # odd hours
"failed_attempts":np.random.randint(8,15,num_anomalies),
"new_device":[1]*num_anomalies,
"ip_changed":[1]*num_anomalies,
"geo_distance_km":np.random.randint(3000,9000,num_anomalies),
"session_duration":np.random.randint(1,5,num_anomalies),
"label":["anomaly"]*num_anomalies
})

df=pd.concat([df,anomalies],ignore_index=True)

print(df.tail(20))

df.to_csv("login_data_with_anomalies.csv",index=False)
print("Anomaly dataset created")