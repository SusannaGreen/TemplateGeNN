import pandas as pd
import plotly
import plotly_express as px

#Upload the data
TemplateBank = pd.read_csv(r'100MassSpinTemplateBank.csv')

chi_eff=[]
for m1, m2, s1, s2 in zip(TemplateBank.mass1.values, TemplateBank.mass2.values, TemplateBank.spin1.values, TemplateBank.spin2.values):
    chi_eff_numerator = m1*s1 + m2*s2
    m_total = m1+m2
    chi_effect = chi_eff_numerator/m_total
    chi_eff.append(chi_effect)

TemplateBank['chi_eff'] = chi_eff

fig = px.scatter_3d(TemplateBank, x='mass1', y='mass2', z='chi_eff')
fig.write_html("100TemplateBank.html")