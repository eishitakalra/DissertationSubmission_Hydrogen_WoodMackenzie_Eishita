from ReadData import read_data
import xpress as xp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from ReadData import interp
from datetime import datetime, timedelta
from ReadData import find_fixed_replace
from ReadData import periods_between
from ReadData import scaled_PPA

def planning_model(P_PPA, df_Grid, df_Elctro, df_Elctro_Costs, df_Battery, df_PPA, Demand, DemandCycle, GreenLimit,  df_lifetime, df_rep_costs, gamma=1, alpha=1.5, beta=1, delta1=1, delta2=1, delta3=1, target_year =2025, target_month = 6,Project_Years = 26, miprelstop = 0.02, maxtime = 120):
    '''
    Solves hydrogen production problem for Wood Mackenzie optimising costs and emissions

    Inputs:
        P_PPA: time series data in dictionary form, indexed by [r,y,m,d,h]. Scaled by capacity of site r which is a parameter in df_PPA, this value can be 1 or a capacity depending on P_PPA data
        df_Grid: DataFrame with hourly CO2 intensity of power from grid and its price
        df_Elctro: DataFrame with Electrolyser parameters for PEM ALK SOEC
        df_Elctro_Costs: DataFrame with CAPEX costs of electrolysers based on different capacities
        df_Battery: DataFrame with Battery storage costs
        df_PPA: Data frame with PPA costs over time horizon and capacity scalar for P_PPA ( NOTE: change to [1,1,1] if P_PPA already scaled)
        Demand: Demand in kg if daily demand assumed 
        Project_Years: The project lifetime in years ( default at 26 for 2025 - 2050 )
        DemandCycle: The time period where demand must be met, if 'Weekly' then weekly constant demand, if 'Daily' then daily constant demand, if 'Monthly' then monthly constant demand
        GreenLimit: The time period across which the green limit is calculated, daily, weekly, monthly or annual
        alpha: the multiplier for PPA price on calculated 'fairprice' based on grid prices DONE
        beta: multiplier on selling price of energy back to the grid DONE
        delta1: multiplier on battery capex DONE
        delta2: multiplier on battery roundtrip efficiency DONE
        delta3: multiplier on battery opex DONE
        df_lifetime: dataframe with lifetime in hours
        df_rep_costs: dataframe for specific electrolyser size data, 2MW, 20MW, or 200MW, for its replacement costs
        gamma: proportion of time elctrolyser assumed to be in operation
        targetyear,targetmonth: first week of which month data is saved and presented graphs for 
        miprelstop: MIP gap at which to terminate (default at 2%)
        maxtime: max time to allow the solver to run for (default 120s)
        
    Outputs:
        Prints MAIN planning data 
        Saves week data
        Shows power graph 
        Shows battery operation 


    '''


    xp.init('/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr')

    prob = xp.problem(name="Hydrogen WoodMac")

    def clean_name(r):
        return r.replace(' ', '_')



        # ------------ SETS -------------
    T = list(
        df_Grid[['Report_Year', 'Report_Month', 'Report_Day', 'Report_Hour']]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    R = list(df_PPA['Renewable Source'].unique())
        
    E = list(df_Elctro['Type'].unique())
        
    days = sorted(set((y, m, d) for (y, m, d, h) in T))
    years = sorted(set(y for (y,m,d,h) in T))   
    months = range(1,13)
    hours = range(1,25)

    df_T = pd.DataFrame(T, columns=['y', 'm', 'd', 'h'])
    days_per_year = df_T.drop_duplicates(['y', 'm', 'd']).groupby('y').size().to_dict()

    def next_hour(y, m, d, h):
        # Convert hour from 1–24 to 0–23
        dt = datetime(y, m, d, h - 1)
        dt_next = dt + timedelta(hours=1)
            # Convert back to 1–24 format
        return (dt_next.year, dt_next.month, dt_next.day, dt_next.hour + 1)

    def prev_hour(y, m, d, h):
        dt = datetime(y, m, d, h - 1)
        dt_prev = dt - timedelta(hours=1)
        dt_prev_hour = dt_prev.hour +1
        return dt_prev.year, dt_prev.month, dt_prev.day, dt_prev_hour
        
    # Index Grid data by (year, month, date, hour)
    df_Grid_index = df_Grid.copy()
    df_Grid_index.set_index(['Report_Year', 'Report_Month', 'Report_Day', 'Report_Hour'],inplace = True)
    df_PPA_index = df_PPA.copy()
    df_PPA_index.set_index('Renewable Source', inplace=True)
    df_Elctro_Costs_index = df_Elctro_Costs.copy()
    df_Elctro_Costs_index.set_index('Technology', inplace=True)

    df_rep_years, df_fixed_rep_costs = find_fixed_replace(gamma,df_Elctro,df_lifetime,df_rep_costs)

    # ------ DECISION VARIABLES ------

    # Proportion of renewable energy contracted to take from renewable site r
    PPA = {(r): xp.var(vartype=xp.continuous, name = f'PPA_{clean_name(r)}', lb= 0, ub=1) for r in R}

    # Power bought from the grid at time t (kW)
    P_Grid_b = {(y,m,d,h): xp.var(vartype=xp.continuous, name = f'P_Grid_b_{y}_{m}_{d}_{h}', lb=0) for (y,m,d,h) in T}

    # Power sold to the grid at time t (kW)
    P_Grid_s = {(y,m,d,h): xp.var(vartype=xp.continuous, name = f'P_Grid_s_{y}_{m}_{d}_{h}',lb=0) for (y,m,d,h) in T}

    # Power taken out of battery at time t (kW)
    P_Bat_out = {(y,m,d,h): xp.var(vartype=xp.continuous, name = f'P_Bat_out_{y}_{m}_{d}_{h}', lb=0) for (y,m,d,h) in T}

    # Power put into battery at time t (kW)
    P_Bat_in = {(y,m,d,h): xp.var(vartype=xp.continuous, name = f'P_Bat_in_{y}_{m}_{d}_{h}',lb=0) for (y,m,d,h) in T}

    # Power put into electrolyser e at time t (kW)
    P_Ez = {(e,y,m,d,h): xp.var(vartype=xp.continuous, name = f'P_Ez_{e}_{y}_{m}_{d}_{h}',lb=0) for e in E for (y,m,d,h) in T}

    # # Power required for putting H2 into storage at time t (kW)
    # P_H2st = {(t): xp.var(vartype=xp.continuous, name = f'P_H2st_{y}_{m}_{d}_{h}') for t in T}

    # # Hydrogen leaving store at time t (kg/h)
    # H_H2st_out = {(t): xp.var(vartype=xp.continuous, name = f'P_H2st_out_{y}_{m}_{d}_{h}') for t in T}

    # # Hydrogen entering store at time t (kg/h)
    # H_H2st_in = {(t): xp.var(vartype=xp.continuous, name = f'P_H2st_in_{t}') for t in T}

    #  Hydrogen leaving electrolyser e at time t (kg/h)
    H_Ez_out = {(e, y,m,d,h): xp.var(vartype=xp.continuous, name = f'H_Ez_out_{e}_{y}_{m}_{d}_{h}',lb=0) for e in E for (y,m,d,h) in T}

    # Energy stored in battery at time t (kWh)
    E_Bat = {(y,m,d,h): xp.var(vartype=xp.continuous, name = f'E_Bat_{y}_{m}_{d}_{h}',lb=0) for (y,m,d,h) in T}

    # # Hydrogen stored at time t (kg)
    # E_H2st = {(t): xp.var(vartype=xp.continuous, name = f'E_H2st_{t}') for t in T}

    # Energy Capacity of battery (kWh)
    Q_Bat_cap = xp.var(vartype=xp.continuous, name='Q_Bat_cap',lb=0)

    # # Energy capacity of H2 storage tank (kg)
    # Q_H2st_cap = xp.var(vartype=xp.continuous, name='Q_H2st_cap')

    # Power capacity of electrolyser e (kW)
    P_Ez_cap = {(e): xp.var(vartype=xp.continuous, name = f'P_Ez_cap_{e}',lb=0) for e in E}

    # Power capacity of battery (kW)
    P_Bat_cap = xp.var(vartype=xp.continuous, name='P_Bat_cap',lb=0)

    # Load factor of electrolyser e at time t
    # Load_Ez = {(e, y,m,d,h): xp.var(vartype=xp.continuous, name = f'Load_Ez_{e}_{y}_{m}_{d}_{h}',lb=0,ub=1) for e in E for (y,m,d,h) in T}

    # Cumulative hours at time t an electrolyser e has been operating for since last stack replacement
    # H_Ez_cum = {(e,y,m,d,h): xp.var(vartype=xp.continuous, name = f'H_Ez_cum_{e}_{y}_{m}_{d}_{h}',lb=0) for e in E for (y,m,d,h) in T}

    # Binary variable for if electrolyser e is on at time t
    # z = {(e, y,m,d,h): xp.var(vartype=xp.binary, name = f'z_{e}_{y}_{m}_{d}_{h}') for e in E for (y,m,d,h) in T}

    # Binary variable for if the battery is being charged or discharged at time t
    # b = {( y,m,d,h): xp.var(vartype=xp.binary, name = f'b_{y}_{m}_{d}_{h}') for (y,m,d,h) in T}

    # Binary variable if stack for electrolyser e is replaced at time t
    # Rep = {(e,y,m,d,h): xp.var(vartype=xp.binary, name = f'R_{e}_{y}_{m}_{d}_{h}') for e in E for (y,m,d,h) in T}

    # Energy needed for electrolyser e per kg of hydrogen output as  a function of the load factor (kWh/kg)
    # Eff_Ez = {(e,y,m,d,h): xp.var(vartype=xp.continuous, name = f'Eff_Ez{e}_{y}_{m}_{d}_{h}',lb=0) for e in E for (y,m,d,h) in T}
        
    # The lifetime of current stack at electrolyser e at time t determined by last replacement 
    # Life_Stack_current = {(e,y,m,d,h): xp.var(vartype=xp.continuous, name = f'Life_Stack_current_{e}_{y}_{m}_{d}_{h}',lb=0) for e in E for (y,m,d,h) in T}

    # Deficit = {(y,m,d,h): xp.var(vartype=xp.continuous, name = f'Def_{y}_{m}_{d}_{h}',lb=0) for (y,m,d,h) in T}

    # Average power from renewables
    PPA_Av = {(r): xp.var(vartype=xp.continuous, name = f'PPA_Av_{r}',lb=0) for r in R}

    # If electrolyser e is built or not
    build = {(e): xp.var(vartype=xp.binary, name = f'build_{e}') for e in E }

    prob.addVariable(PPA, P_Grid_b, P_Grid_s, P_Bat_out, P_Bat_in, P_Ez, H_Ez_out, E_Bat, Q_Bat_cap, P_Ez_cap, P_Bat_cap,PPA_Av,build)

    # --------- INDEX DATA ------------

    # Index electrolyser data by electrolyser name
    df_Elctro_index = df_Elctro.copy()
    df_Elctro_index.set_index('Type', inplace=True)

    # Find day indexes
    day_index = {}
    counter = 0
    for (y, m, d, h) in sorted(T):  
        if (y, m, d) not in day_index:
            day_index[(y, m, d)] = counter
            counter += 1

    # --------- PARAMETERS ------------

    #  Power available from renewable site r at time t (kW) = P_PPA

    # Capacity size as a scale for each renewable r (kW) used if given data needed to be normalised and scaled for capacity
    scale = df_PPA_index['Renewable Capacity Scale (kW)'].to_dict()

    # Constant on-site daily hydrogen demand (kg)
    D_H2 = float(Demand)

    # Round-trip efficiency for battery
    Eff_Bat = float(df_Battery["Round trip efficiency"].iloc[0])

    # Electrolyser Simple efficiency
    Eff_Ez = df_Elctro_index['Simple Efficiency'].to_dict()

    # CO2 Intensity of power from the grid at time t (kg/kWh)
    Int_Grid = df_Grid_index['CO2 Intensity (kg CO2/kWh)'].to_dict()

    # Maximum CO2 emissions in kg CO2 e/kg H2
    Int_max = 2.4

    # Min and Max load for electrolyser e
    Ez_min_load = df_Elctro_index['Minimum Load'].to_dict()
    Ez_max_load = df_Elctro_index['Maximum Load'].to_dict()

    # Duration in hours of the battery
    Bat_dur = float(df_Battery['Duration (hrs)'].iloc[0])

    # Number of years over time horizon
    N_years = len(years)

    # Stack lifetime for electrolyser e in hours if replaced/built in time period t, have to copy all years for all m,d,h
    # Life_Stack = df_lifetime_full

    # System efficiency degredation 
    # x_eff_deg = df_Elctro_index['System Efficiency Degradation'].to_dict()

    # System degredation factor 
    # x_deg_fact = {}
    # for e in E:
    #     for (y,m,d,h) in T:
    #         y_int = int(y)
    #         x_deg_fact[(e,y,m,d,h)] = x_eff_deg[e]**(x_FID - y_int)

    # Cost for buying and selling power from the grid
    C_Grid = df_Grid_index['Price (£/kWh, real 2025)'].to_dict()

    C_PPA = df_PPA_index['PPA Price (£/kWh)'].to_dict()

    # CAPEX Cost for capacity of battery store
    C_Bat_capex = float(df_Battery['Capex (£/kWh)'].iloc[0])

    # CAPEX Cost for electrolyser e
    C_Ez_capex = df_Elctro_Costs_index['Total Installed Cost (TIC) (£/kW)'].to_dict()

    # Fixed OPEX cost for electrolyser e as proportion of capex
    C_Ez_fixed_opex = df_Elctro_index['Fixed Opex percent'].to_dict()

    # Fixed OPEX cost for battery as proportion of capex
    C_Bat_fixed_opex = float(df_Battery['Fixed Opex percent'].iloc[0])

    # Replacement costs of the stack in electrolyser e if replaced in time period t
    # C_Replace = df_replacement_full

    # Costs which make up CAPEX cost of electrolysers
    C_Ez_BoS = df_Elctro_Costs_index['Balance of Stack (£/kW)'].to_dict()
    C_Ez_BoP = df_Elctro_Costs_index['Balance of Plant (£/kW)'].to_dict()
    C_Ez_EPC = df_Elctro_Costs_index['Engineering, Procurement & Construction costs (£/kW)'].to_dict()
    C_Ez_Owners = df_Elctro_Costs_index['Owners costs (£/kW)'].to_dict()

    # Fixed replacement costs
    C_Replace = {}
    for e in E:
        C_Replace[e] = float(df_fixed_rep_costs[e].iloc[0])

    # ---------- CONSTRAINTS ------------

    for t in T:
        prob.addConstraint(P_Grid_b[t] >= 0)
        prob.addConstraint(P_Grid_s[t] >= 0)

    # Power Balance:
    prob.addConstraint( xp.Sum(PPA[r]*scale[r]*P_PPA[(r,y,m,d,h)] for r in R )+ P_Grid_b[(y,m,d,h)]+ P_Bat_out[(y,m,d,h)] == P_Grid_s[(y,m,d,h)] + P_Bat_in[(y,m,d,h)] + xp.Sum(P_Ez[(e,y,m,d,h)] for e in E) for (y,m,d,h) in T)
        
    # prob.addConstraint( Deficit[y,m,d,h] >= xp.Sum(P_Ez[e,y,m,d,h] for e in E) - xp.Sum(P_PPA[r,y,m,d,h] for r in R) - P_Bat_out[y,m,d,h] for (y,m,d,h) in T )
    # prob.addConstraint(P_Grid_b[y,m,d,h] <= Deficit[y,m,d,h] for (y,m,d,h) in T)
    # # prob.addConstraint( P_Grid_s[y,m,d,h] <=  xp.Sum(PPA[r]*P_PPA[(r,y,m,d,h)] for r in R) for (y,m,d,h) in T)
    # prob.addConstraint( P_Grid_s[y,m,d,h] <=  xp.Sum(PPA[r]*P_PPA[(r,y,m,d,h)] for r in R ) for (y,m,d,h) in T)

    # Limit on PPA
    # P_Ez_Avg = max(Eff_Ez.values())*Demand/24
    N_periods = len(T)
    # for r in R:
    #     prob.addConstraint( PPA_Av[r] == xp.Sum( P_PPA[(r,y,m,d,h)] for (y,m,d,h) in T )/N_periods )

    # prob.addConstraint( PPA[r]*PPA_Av[r] <= P_Ez_Avg for r in R)

    
    # Hydrogen Balance:
    if DemandCycle == 'Daily':
        for (y,m,d) in days:
            prob.addConstraint( xp.Sum(H_Ez_out[(e,y,m,d,h)] for e in E for h in hours) ==  D_H2 )
    
    if DemandCycle == 'Weekly':
        unique_weeks = set()
        
        for (y, m, d, h) in T:
            week = day_index[(y, m, d)] // 7
            unique_weeks.add(week)
        
        for week in unique_weeks:
            prob.addConstraint( xp.Sum( H_Ez_out[(e, y, m, d, h)] for e in E for (y, m, d, h) in T if (day_index[(y, m, d)] // 7) == week) == 7 * D_H2 )

    if DemandCycle == 'Monthly':
        unique_months = set((y, m) for (y, m, d, h) in T)
        
        for (y, m) in unique_months:
            days_in_month = len({d for (yy, mm, d, h) in T if yy == y and mm == m})

            prob.addConstraint( xp.Sum( H_Ez_out[(e, yy, mm, dd, hh)] for e in E for (yy, mm, dd, hh) in T if yy == y and mm == m) == days_in_month * D_H2)
    
    # Capacity of Power bought from grid, sold to gris and taken from PPA

    # prob.addConstraint( xp.Sum(P_PPA[r,y,m,d,h]*PPA[r] for r in R for h in hours)/24 <=  5*(xp.Sum(P_Ez_cap[e] for e in E)) for (y,m,d) in days)
    # prob.addConstraint( xp.Sum(P_Grid_b[y,m,d,h] for h in hours)/24 <=  5*(xp.Sum(P_Ez_cap[e] for e in E)) for (y,m,d) in days)

    # Battery:
    for (y, m, d, h) in T:
        t_next = next_hour(y, m, d, h)
        if t_next in T:
            prob.addConstraint( E_Bat[t_next] == E_Bat[(y, m, d, h)] + delta2*Eff_Bat * P_Bat_in[(y, m, d, h)] - P_Bat_out[(y, m, d, h)])
    prob.addConstraint( 0 <= P_Bat_out[t] for t in T )
    prob.addConstraint( 0 <= P_Bat_in[t] for t in T )
    prob.addConstraint( P_Bat_in[t] <= P_Bat_cap for t in T )
    prob.addConstraint( P_Bat_out[t] <= P_Bat_cap for t in T )
    prob.addConstraint( 0 <= E_Bat[t] for t in T )
    prob.addConstraint( E_Bat[t]<= Q_Bat_cap for t in T )
    prob.addConstraint( Q_Bat_cap == Bat_dur*P_Bat_cap )

    # prob.addConstraint( P_Bat_in[t] <= 9999999999999*(1-b[t]) for t in T)
    # prob.addConstraint( P_Bat_out[t] <= 9999999999999*b[t] for t in T)
        
    # Average CO2 Emissions:
    if GreenLimit == 'Yearly':
        for y in years:
            prob.addConstraint(xp.Sum( ( Int_Grid[(y_t,m,d,h)]*(P_Grid_b[(y_t,m,d,h)]))/(days_per_year[y_t]*D_H2) for (y_t,m,d,h) in T if y_t == y) <= Int_max)

    if GreenLimit == 'Monthly':
        unique_months = set((y_t, m) for (y_t, m, d, h) in T)
        for (y, m) in unique_months:
            days_in_month = len({d for (yy, mm, d, h) in T if yy == y and mm == m})

            prob.addConstraint(xp.Sum((Int_Grid[(yy, mm, dd, hh)] * P_Grid_b[(yy, mm, dd, hh)]) / (days_in_month * D_H2) for (yy, mm, dd, hh) in T if yy == y and mm == m) <= Int_max)

    if GreenLimit == 'Weekly':
        unique_weeks = set()

        for (y, m, d, h) in T:
            week = day_index[(y, m, d)] // 7
            unique_weeks.add(week)
        
        for week in unique_weeks:
            prob.addConstraint(xp.Sum( ( Int_Grid[(y,m,d,h)]*(P_Grid_b[(y,m,d,h)]))/(7*D_H2) for (y,m,d,h) in T if (day_index[(y, m, d)] // 7) == week) <= Int_max)


    # Electrolysers:
    # prob.addConstraint(Eff_Ez[(e,y,m,d,h)] == ( (x5[e]*Load_Ez[(e,y,m,d,h)])**5 + (x4[e]*Load_Ez[(e,y,m,d,h)])**4 + (x3[e]*Load_Ez[(e,y,m,d,h)])**3 + (x2[e]*Load_Ez[(e,y,m,d,h)])**2 + (x1[e]*Load_Ez[(e,y,m,d,h)]) + x0[e])*x_deg_fact[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)
    prob.addConstraint( P_Ez[(e,y,m,d,h)] == H_Ez_out[(e,y,m,d,h)]*Eff_Ez[e] for e in E for (y,m,d,h) in T)

    prob.addConstraint( 0 <= P_Ez[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)

    # prob.addConstraint( P_Bat_cap <= 10*xp.Sum(P_Ez_cap[e] for e in E) )

    # NONLINEAR:
    # prob.addConstraint( H_Ez_out[(e,y,m,d,h)] <= z[(e,y,m,d,h)]*9999999999999 for e in E for (y,m,d,h) in T)
    # prob.addConstraint( P_Ez[(e,y,m,d,h)] <= P_Ez_cap[e] for e in E for (y,m,d,h) in T)

    # prob.addConstraint( P_Ez[(e,y,m,d,h)] <= z[(e,y,m,d,h)]*Ez_max_load[e]*P_Ez_cap[e] for e in E for (y,m,d,h) in T)
    # prob.addConstraint( z[(e,y,m,d,h)]*Ez_min_load[e]*P_Ez_cap[e] <= P_Ez[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)

    # Linear
    # prob.addConstraint( P_Ez[(e,y,m,d,h)] <= Ez_max_load[e]*P_Ez_cap[e] for e in E for (y,m,d,h) in T)

    Constraint_Capacity = {}
    for e in E:
        Constraint_Capacity[(e,y,m,d,h)] = prob.addConstraint( P_Ez[(e,y,m,d,h)] <= P_Ez_cap[e] for (y,m,d,h) in T)
        Constraint_Capacity[(e,y,m,d,h)]

    # prob.addConstraint(P_Ez_cap[e]*Load_Ez[(e,y,m,d,h)] == P_Ez[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)

    # prob.addConstraint( z[(e,y,m,d,h)]<= P_Ez_cap[e] for e in E for (y,m,d,h) in T)
    # prob.addConstraint( Ez_min_load[e]*z[(e,y,m,d,h)]<= Load_Ez[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)
    # prob.addConstraint( Load_Ez[(e,y,m,d,h)] <= Ez_min_load[e]*z[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)
    # prob.addConstraint( z[(e,y,m,d,h)]<= P_Ez[(e,y,m,d,h)]*9999999999999 for e in E for (y,m,d,h) in T)


    # prob.addConstraint(P_Ez[(e,y,m,d,h)] >= Ez_min_load[e] * P_Ez_cap[e] * z[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)

    # # Stack Replacement:
    # prob.addConstraint( Life_Stack_current[(e,2025,1,1,1)] == df_lifetime_full[(e,2025,1,1,1)] for e in E)
    # for (y, m, d, h) in T:
    #     y_prev, m_prev, d_prev, h_prev = prev_hour(y, m, d, h)
    #     if (y_prev,m_prev,d_prev,h_prev) in T:
    #         for e in E:
    #             prob.addConstraint( Life_Stack_current[(e,y,m,d,h)] == Rep[(e,y,m,d,h)]*(df_lifetime_full[(e,y,m,d,h)]) + (1 - Rep[(e,y,m,d,h)])*(Life_Stack_current[(e,y_prev,m_prev,d_prev,h_prev)])  )

    # prob.addConstraint(H_Ez_cum[(e,y,m,d,h)] == (1-Rep[(e,y,m,d,h)])*(H_Ez_cum[(e,y,m,d,h)] + z[(e,y,m,d,h)]) for e in E for (y,m,d,h) in T )

    # prob.addConstraint(H_Ez_cum[(e,y,m,d,h)] <= Life_Stack_current[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)
    for e in E:
        prob.addConstraint( P_Ez_cap[e] <= 200000000*build[e] )

    # ---------- OBJECTIVE FUNCTION ----------

    # CAPEX Costs:
    CAPEX = delta1*C_Bat_capex*P_Bat_cap*Bat_dur + xp.Sum(P_Ez_cap[e]*C_Ez_capex[e] for e in E)
    Hourly_CAPEX = CAPEX/227903

    # PPA Costs
    PPA_Cost = xp.Sum(alpha*scale[r]*P_PPA[(r,y,m,d,h)]*C_PPA[r]*PPA[r] for r in R for (y,m,d,h) in T) 
    Av_Hourly_PPA_Cost = PPA_Cost/N_periods

    # OPEX Costs:
    # Electrolyser and Battery fixed OPEX based on % of CAPEX
    fixed_OPEX = 26*( delta3*C_Bat_fixed_opex*C_Bat_capex*P_Bat_cap*Bat_dur + xp.Sum( C_Ez_fixed_opex[e]*P_Ez_cap[e]*C_Ez_capex[e] for e in E))
    Av_Hourly_OPEX = fixed_OPEX/227903

    # Variable OPEX for Power bought and sold on Grid
    variable_OPEX = xp.Sum(C_Grid[(y,m,d,h)]*( P_Grid_b[(y,m,d,h)] - beta*P_Grid_s[(y,m,d,h)]) for (y,m,d,h) in T)
    Av_Hourly_variable_OPEX = variable_OPEX/N_periods

    # Fixed Stack Replacement costs
    stack_CAPEX = xp.Sum(C_Replace[e]*build[e] for e in E)
    Hourly_stack_rep_CAPEX = stack_CAPEX/227903

    Total_costs = CAPEX + fixed_OPEX + variable_OPEX 
    prob.setObjective(Hourly_CAPEX+Av_Hourly_OPEX+Av_Hourly_variable_OPEX+Av_Hourly_PPA_Cost+ Hourly_stack_rep_CAPEX, sense = xp.minimize)
    prob.controls.miprelstop = miprelstop
    prob.controls.maxtime = maxtime
    prob.setControl('TIMELIMIT',maxtime)
    prob.solve()

    # ------- PRINT RESULTS ------------

    # Printing results into a text file
    with open("planning_output.txt", "w") as f:

        print("Total costs:",file=f)
        print("Total CAPEX Costs: £", prob.getSolution(CAPEX),file=f)
        print("Total Fixed OPEX: £", prob.getSolution(fixed_OPEX),file=f)
        print("Hourly Costs:",file=f)
        print("Average Hourly PPA cost: £", prob.getSolution(Av_Hourly_PPA_Cost),file=f)
        # print("variable_OPEX: £", prob.getSolution(variable_OPEX))
        # print("Total costs £", prob.getSolution(Total_costs))
        print("Hourly CAPEX £",prob.getSolution(Hourly_CAPEX),file=f)
        print("Hourly FIXED OPEX £",prob.getSolution(Av_Hourly_OPEX),file=f)
        print("Average Hourly Power Costs £",prob.getSolution(Av_Hourly_variable_OPEX),file=f)
        print("Total Hourly Costs £",prob.getSolution(Hourly_CAPEX+Av_Hourly_OPEX+Av_Hourly_variable_OPEX+Hourly_stack_rep_CAPEX+Av_Hourly_PPA_Cost),file=f)
        print("Planning Decisions:",file=f)
        print("Battery capacity is ", prob.getSolution(P_Bat_cap),"kW",file=f)
        for e in E:
            print("Electrolyser ",e, " has capacity ", prob.getSolution(P_Ez_cap[e]),"kW ",file=f)
        for  r in R:
            print("PPA size for ",r," is ",prob.getSolution(PPA[r]),file=f)


    # Printing results in terminal for checking solutions
    print("Total costs:")
    print("Total CAPEX Costs: £", prob.getSolution(CAPEX))
    print("Total Fixed OPEX: £", prob.getSolution(fixed_OPEX))
    print("Hourly Costs:")
    print("Average Hourly PPA cost: £", prob.getSolution(Av_Hourly_PPA_Cost))
    print("Hourly CAPEX £",prob.getSolution(Hourly_CAPEX))
    print("Hourly FIXED OPEX £",prob.getSolution(Av_Hourly_OPEX))
    print("Average Hourly Power Costs £",prob.getSolution(Av_Hourly_variable_OPEX))
    print("Total Hourly Costs £",prob.getSolution(Hourly_CAPEX+Av_Hourly_OPEX+Av_Hourly_variable_OPEX+Hourly_stack_rep_CAPEX+Av_Hourly_PPA_Cost))
    print("Planning Decisions:")
    print("Battery capacity is ", prob.getSolution(P_Bat_cap),"kW")
    for e in E:
        print("Electrolyser ",e, " has capacity ", prob.getSolution(P_Ez_cap[e]),"kW ")
    for  r in R:
        print("PPA size for ",r," is ",prob.getSolution(PPA[r]))


    print("Power bought from grid on day 1 is ", prob.getSolution(xp.Sum(P_Grid_b[2025,1,1,h] for h in hours)))
    print("Power sold to grid on day 1 is ", prob.getSolution(xp.Sum(P_Grid_s[2025,1,1,h] for h in hours)))

    P_Ez_sol = {(e,y,m,d,h): prob.getSolution(P_Ez[(e,y,m,d,h)]) for (e,y,m,d,h) in P_Ez if m == target_month if y == target_year}
    P_Bat_in_sol = {(y,m,d,h): prob.getSolution(P_Bat_in[(y,m,d,h)]) for (y,m,d,h) in P_Bat_in if m == target_month if y == target_year}
    P_Bat_out_sol = {(y,m,d,h): prob.getSolution(P_Bat_out[(y,m,d,h)]) for (y,m,d,h) in P_Bat_out if m == target_month if y == target_year}
    P_Grid_s_sol = {(y,m,d,h): prob.getSolution(P_Grid_s[(y,m,d,h)]) for (y,m,d,h) in P_Grid_s if m == target_month if y == target_year}
    P_Grid_b_sol = {(y,m,d,h): prob.getSolution(P_Grid_b[(y,m,d,h)])for (y,m,d,h) in P_Grid_b if m == target_month if y == target_year}
    P_PPA_sol = {(r,y,m,d,h): prob.getSolution(scale[r]*P_PPA[(r,y,m,d,h)]*PPA[r]) for (r,y,m,d,h) in P_PPA if m == target_month if y == target_year}
    E_Bat_sol = {(y,m,d,h) : prob.getSolution(E_Bat[(y,m,d,h)]) for (y,m,d,h) in E_Bat if m == target_month if y == target_year}
    ppa = {(r,y,m,d,h): P_PPA[(r,y,m,d,h)] for (r,y,m,d,h) in P_PPA if m == target_month if y == target_year}
    Av_hr_yr_P_Ez_sol = {(e,yy): prob.getSolution( xp.Sum(P_Ez[(e,y,m,d,h)] for (y,m,d,h) in T if y == yy)/(24*days_per_year[y]) ) for e in E for yy in years}
    Av_hr_yr_P_Bat_in_sol = {(yy): prob.getSolution( xp.Sum(P_Bat_in[(y,m,d,h)] for (y,m,d,h) in T if y == yy)/(24*days_per_year[y]) )for yy in years}
    Av_hr_yr_P_Bat_out_sol = {(yy): prob.getSolution( xp.Sum(P_Bat_out[(y,m,d,h)] for (y,m,d,h) in T if y == yy)/(24*days_per_year[y]) ) for yy in years}

    PPA_sol = {(r) : prob.getSolution(PPA[r]) for r in R}
    Ez_cap_sol = {(e) : prob.getSolution(P_Ez_cap[e]) for e in E}
    Bat_cap_sol = prob.getSolution(P_Bat_cap)
    
    # Build base DataFrame with all unique time points
    all_times = set(P_Bat_in_sol.keys()) | set(P_Bat_out_sol.keys()) | set(P_Grid_b_sol.keys()) | set(P_Grid_s_sol.keys())
    all_times |= { (y, m, d, h) for (e, y, m, d, h) in P_Ez_sol }
    all_times |= { (y, m, d, h) for (r, y, m, d, h) in P_PPA_sol }
    all_times |= { (y, m, d, h) for (r, y, m, d, h) in ppa }

    # Convert to datetime index 
    dt_index = pd.to_datetime([f"{y}-{m:02d}-{d:02d} {h-1:02d}:00" for (y, m, d, h) in all_times])
    plot_df = pd.DataFrame(index=dt_index)

    # # Add scalar power flows
    def insert_solution_column(df, sol_dict, column_name):
        s = pd.Series(sol_dict)
        s.index = pd.to_datetime([f"{y}-{m:02d}-{d:02d} {h-1:02d}:00" for (y,m,d,h) in s.index])
        s = s.sort_index()
        df[column_name] = s
    insert_solution_column(plot_df, P_Grid_b_sol, 'Bought from Grid')
    insert_solution_column(plot_df, P_Grid_s_sol, 'Sold to Grid')
    insert_solution_column(plot_df, P_Bat_in_sol, 'Battery in')
    insert_solution_column(plot_df, P_Bat_out_sol, 'Battery out')
    insert_solution_column(plot_df, E_Bat_sol, 'Energy in Battery')

    for e in E:
        e_data = { (y, m, d, h): val for (ee, y, m, d, h), val in P_Ez_sol.items() if ee == e }
        s = pd.Series(e_data)
        s.index = pd.to_datetime([f"{y}-{m:02d}-{d:02d} {h-1:02d}:00" for (y, m, d, h) in s.index])
        s = s.sort_index()
        plot_df[f'Electrolyser {e}'] = s


    for r in R:
        r_data = { (y, m, d, h): val for (rr, y, m, d, h), val in P_PPA_sol.items() if rr == r }
        s = pd.Series(r_data)
        s.index = pd.to_datetime([f"{y}-{m:02d}-{d:02d} {h-1:02d}:00" for (y, m, d, h) in s.index])
        s = s.sort_index()
        plot_df[f'{r} PPA'] = s

    # for r in R:
    #     r_data = { (y, m, d, h): val for (rr, y, m, d, h), val in ppa.items() if rr == r }
    #     s = pd.Series(r_data)
    #     s.index = pd.to_datetime([f"{y}-{m:02d}-{d:02d} {h-1:02d}:00" for (y, m, d, h) in s.index])
    #     s = s.sort_index()
    #     plot_df[f'ppa_{r}'] = s


    # # Sort index
    plot_df = plot_df.sort_index()

    # # Filter to a week
    plot_df = plot_df.loc[plot_df.index[0]:plot_df.index[0] + pd.Timedelta(days=7)]

    # # Plot each column on its own subplot
    fig, axs = plt.subplots(len(plot_df.columns), 1, figsize=(14, 2.5 * len(plot_df.columns)), sharex=True)

    for i, col in enumerate(plot_df.columns):
        axs[i].plot(plot_df.index, plot_df[col])
        axs[i].set_ylabel(col)
        axs[i].grid(True)

    axs[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()

    plot_df = plot_df.loc[plot_df.index[0]:plot_df.index[0] + pd.Timedelta(days=31)]

    plt.figure(figsize=(16, 6))

    for col in plot_df.columns:
        if col != 'Energy in Battery':
            plt.plot(plot_df.index, plot_df[col], label=col)

    plt.title("Power Flows Over An Example Week")
    plt.ylabel("Power (kW)")
    plt.xlabel("Time")
    plt.legend(loc='upper right', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return PPA_sol, Ez_cap_sol, Bat_cap_sol, P_Ez_sol, P_Bat_in_sol, P_Bat_out_sol, Av_hr_yr_P_Ez_sol, Av_hr_yr_P_Bat_in_sol, Av_hr_yr_P_Bat_out_sol



# function input set of years & set of months & full df_RE and df_Grid & cap bat & cap ez & ppas & gamma
# find penalty based on gamma
# create a list
# create a counter at 0
# loop over a set of years
# loop over set of months
# df_RE1 = ...from df_RE & grid
# -----solve the optim problem over week: has unit commit, min n max if on, const eff, penalty 
# -----print for each one how it operates label by month name
# add to list the value ehich is number of hrs on /168
# add one to counter for each week
# end both loops find value which is sum of all values in list div by counter
# return the value this becomes new gamma

# do individually gamma func, set new gamma, func , set new gamma, func iterative

def operational_model(P_PPA,df_Grid, df_Elctro, df_Elctro_Costs,df_Battery,df_PPA,Demand,DemandCycle,GreenLimit,df_lifetime,df_rep_costs,PPA_sol, Ez_cap_sol, Bat_cap_sol, gamma=1,alpha=1.5,beta=1,Project_Years=26,miprelstop=0.02,maxtime=600):
    '''
    The model which solves hydrogen production problem for Wood Mackenzie optimising costs and emissions
    It includes unit commitment and finds when the elctrolyser is on and off, hence includes min/max load and replacment cost as a penalty
    MINLP which has a penalty for electrolyser being on, calculated as a fixed cost using df_lifetime and df_rep_costs assuming the elctrolyser was on gamma*100% of the time
    Assumes constant efficiency of electrolysers
    Solves assuming the data inputted is only for a week

    Inputs:
        P_PPA:
        df_Grid: DataFrame with hourly CO2 intensity of power from grid and its price for a week of dat
        df_Elctro: DataFrame with Electrolyser parameters for PEM ALK SOEC
        df_Elctro_Costs: DataFrame with CAPEX costs of electrolysers based on different capacities
        df_Battery: DataFrame with Battery storage costs
        Demand: Demand in kg if daily demand assumed 
        Project_Years: The project lifetime in years ( default at 26 for 2025 - 2050 )
        DemandCycle: The time period where demand must be met, if 'Weekly' then weekly constant demand, if 'Daily' then daily constant demand, if 'Monthly' then monthly constant demand
        GreenLimit: The time period across which the green limit is calculated, daily, weekly, monthly or annual
        alpha: the multiplier for PPA price on calculated 'fairprice' based on grid prices 
        beta: multiplier on selling price of energy back to the grid 
        df_lifetime: lifetime dataframe
        df_rep_costs: replacement costs dataframe
        gamma: proportion of time elctrolyser assumed to be on
        PPA_sol: PPA capacity
        Bat_cap_sol: battery capacity
        Ez_cap_sol: electrolyser capacity
        targetyear,targetmonth: first week of which month data is saved and presented graphs for 
        miprelstop: MIP gap at which to terminate (default at 2%)
        maxtime: max time to allow the solver to run for (default 120s)
    
    Outputs:
        Plots operation over the time horizon the model is run for
        Gives the number of hours the electrolyser is on for over the time horizon


    ''' 
    xp.init('/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr')

    prob = xp.problem(name="Hydrogen WoodMac")

    def clean_name(r):
        return r.replace(' ', '_')



    # ------------ SETS -------------
    T = list(
        df_Grid[['Report_Year', 'Report_Month', 'Report_Day', 'Report_Hour']]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
        
    E = list(df_Elctro['Type'].unique())
    R = list(df_PPA['Renewable Source'].unique())
        
    days = sorted(set((y, m, d) for (y, m, d, h) in T))
    years = sorted(set(y for (y,m,d,h) in T))   
    months = range(1,13)
    hours = range(1,25)

    df_T = pd.DataFrame(T, columns=['y', 'm', 'd', 'h'])
    days_per_year = df_T.drop_duplicates(['y', 'm', 'd']).groupby('y').size().to_dict()

    def next_hour(y, m, d, h):
        # Convert hour from 1–24 to 0–23
        dt = datetime(y, m, d, h - 1)
        dt_next = dt + timedelta(hours=1)
            # Convert back to 1–24 format
        return (dt_next.year, dt_next.month, dt_next.day, dt_next.hour + 1)

    def prev_hour(y, m, d, h):
        dt = datetime(y, m, d, h - 1)
        dt_prev = dt - timedelta(hours=1)
        dt_prev_hour = dt_prev.hour +1
        return dt_prev.year, dt_prev.month, dt_prev.day, dt_prev_hour
        
    # Index Grid data by (year, month, date, hour)
    df_Grid_index = df_Grid.copy()
    df_Grid_index.set_index(['Report_Year', 'Report_Month', 'Report_Day', 'Report_Hour'],inplace = True)
    df_PPA_index = df_PPA.copy()
    df_PPA_index.set_index('Renewable Source', inplace=True)
    df_Elctro_Costs_index = df_Elctro_Costs.copy()
    df_Elctro_Costs_index.set_index('Technology', inplace=True)

    df_rep_years, df_fixed_rep_costs = find_fixed_replace(gamma,df_Elctro,df_lifetime,df_rep_costs)

    # ------ DECISION VARIABLES ------

    # Proportion of renewable energy contracted to take from renewable site r
    PPA = {(r): xp.var(vartype=xp.continuous, name = f'PPA_{clean_name(r)}', lb= 0, ub=1) for r in R}

    # Power bought from the grid at time t (kW)
    P_Grid_b = {(y,m,d,h): xp.var(vartype=xp.continuous, name = f'P_Grid_b_{y}_{m}_{d}_{h}', lb=0) for (y,m,d,h) in T}

    # Power sold to the grid at time t (kW)
    P_Grid_s = {(y,m,d,h): xp.var(vartype=xp.continuous, name = f'P_Grid_s_{y}_{m}_{d}_{h}',lb=0) for (y,m,d,h) in T}

    # Power taken out of battery at time t (kW)
    P_Bat_out = {(y,m,d,h): xp.var(vartype=xp.continuous, name = f'P_Bat_out_{y}_{m}_{d}_{h}', lb=0) for (y,m,d,h) in T}

    # Power put into battery at time t (kW)
    P_Bat_in = {(y,m,d,h): xp.var(vartype=xp.continuous, name = f'P_Bat_in_{y}_{m}_{d}_{h}',lb=0) for (y,m,d,h) in T}

    # Power put into electrolyser e at time t (kW)
    P_Ez = {(e,y,m,d,h): xp.var(vartype=xp.continuous, name = f'P_Ez_{e}_{y}_{m}_{d}_{h}',lb=0) for e in E for (y,m,d,h) in T}

    # # Power required for putting H2 into storage at time t (kW)
    # P_H2st = {(t): xp.var(vartype=xp.continuous, name = f'P_H2st_{y}_{m}_{d}_{h}') for t in T}

    # # Hydrogen leaving store at time t (kg/h)
    # H_H2st_out = {(t): xp.var(vartype=xp.continuous, name = f'P_H2st_out_{y}_{m}_{d}_{h}') for t in T}

    # # Hydrogen entering store at time t (kg/h)
    # H_H2st_in = {(t): xp.var(vartype=xp.continuous, name = f'P_H2st_in_{t}') for t in T}

    #  Hydrogen leaving electrolyser e at time t (kg/h)
    H_Ez_out = {(e, y,m,d,h): xp.var(vartype=xp.continuous, name = f'H_Ez_out_{e}_{y}_{m}_{d}_{h}',lb=0) for e in E for (y,m,d,h) in T}

    # Energy stored in battery at time t (kWh)
    E_Bat = {(y,m,d,h): xp.var(vartype=xp.continuous, name = f'E_Bat_{y}_{m}_{d}_{h}',lb=0) for (y,m,d,h) in T}

    # # Hydrogen stored at time t (kg)
    # E_H2st = {(t): xp.var(vartype=xp.continuous, name = f'E_H2st_{t}') for t in T}

    # # Energy capacity of H2 storage tank (kg)
    # Q_H2st_cap = xp.var(vartype=xp.continuous, name='Q_H2st_cap')

    # Load factor of electrolyser e at time t
    # Load_Ez = {(e, y,m,d,h): xp.var(vartype=xp.continuous, name = f'Load_Ez_{e}_{y}_{m}_{d}_{h}',lb=0,ub=1) for e in E for (y,m,d,h) in T}

    # Binary variable for if electrolyser e is on at time t
    z = {(e, y,m,d,h): xp.var(vartype=xp.binary, name = f'z_{e}_{y}_{m}_{d}_{h}') for e in E for (y,m,d,h) in T}

    # Energy needed for electrolyser e per kg of hydrogen output as  a function of the load factor (kWh/kg)
    # Eff_Ez = {(e,y,m,d,h): xp.var(vartype=xp.continuous, name = f'Eff_Ez{e}_{y}_{m}_{d}_{h}',lb=0) for e in E for (y,m,d,h) in T}

    # If electrolyser e is built or not
    # build = {(e): xp.var(vartype=xp.binary, name = f'build_{e}') for e in E }

    prob.addVariable(PPA, P_Grid_b, P_Grid_s, P_Bat_out, P_Bat_in, P_Ez, H_Ez_out, E_Bat,z)

    # --------- INDEX DATA ------------

    # Index electrolyser data by electrolyser name
    df_Elctro_index = df_Elctro.copy()
    df_Elctro_index.set_index('Type', inplace=True)

    # Find day indexes
    day_index = {}
    counter = 0
    for (y, m, d, h) in sorted(T):  
        if (y, m, d) not in day_index:
            day_index[(y, m, d)] = counter
            counter += 1

    # --------- PARAMETERS ------------

    #  Power available from renewable site r at time t (kW) = P_PPA

    # Capacity size as a scale for each renewable r (kW) used if given data needed to be normalised and scaled for capacity
    scale = df_PPA_index['Renewable Capacity Scale (kW)'].to_dict()

    # Constant on-site daily hydrogen demand (kg)
    D_H2 = float(Demand)

    # Round-trip efficiency for battery
    Eff_Bat = float(df_Battery["Round trip efficiency"].iloc[0])

    # Electrolyser Simple efficiency
    Eff_Ez = df_Elctro_index['Simple Efficiency'].to_dict()

    # CO2 Intensity of power from the grid at time t (kg/kWh)
    Int_Grid = df_Grid_index['CO2 Intensity (kg CO2/kWh)'].to_dict()

    # Maximum CO2 emissions in kg CO2 e/kg H2
    Int_max = 2.4

    # Min and Max load for electrolyser e
    Ez_min_load = df_Elctro_index['Minimum Load'].to_dict()
    Ez_max_load = df_Elctro_index['Maximum Load'].to_dict()

    # Duration in hours of the battery
    Bat_dur = float(df_Battery['Duration (hrs)'].iloc[0])

    # Number of years over time horizon
    N_years = len(years)
    N_periods = len(T)

    # Final Investment Decison Year
    x_FID = 2025

    # System efficiency degredation 
    x_eff_deg = df_Elctro_index['System Efficiency Degradation'].to_dict()

    # System degredation factor 
    x_deg_fact = {}
    for e in E:
        for (y,m,d,h) in T:
            y_int = int(y)
            x_deg_fact[(e,y,m,d,h)] = x_eff_deg[e]**(x_FID - y_int)

    # Coefficients for linear efficiency
    mle = df_Elctro_index['m'].to_dict()
    cle = df_Elctro_index['c'].to_dict()

    # Cost for buying and selling power from the grid
    C_Grid = df_Grid_index['Price (£/kWh, real 2025)'].to_dict()

    # CAPEX Cost for capacity of battery store
    C_Bat_capex = float(df_Battery['Capex (£/kWh)'].iloc[0])

    # CAPEX Cost for electrolyser e
    C_Ez_capex = df_Elctro_Costs_index['Total Installed Cost (TIC) (£/kW)'].to_dict()

    # Fixed OPEX cost for electrolyser e as proportion of capex
    C_Ez_fixed_opex = df_Elctro_index['Fixed Opex percent'].to_dict()

    # Fixed OPEX cost for battery as proportion of capex
    C_Bat_fixed_opex = float(df_Battery['Fixed Opex percent'].iloc[0])

    # Costs which make up CAPEX cost of electrolysers
    C_Ez_BoS = df_Elctro_Costs_index['Balance of Stack (£/kW)'].to_dict()
    C_Ez_BoP = df_Elctro_Costs_index['Balance of Plant (£/kW)'].to_dict()
    C_Ez_EPC = df_Elctro_Costs_index['Engineering, Procurement & Construction costs (£/kW)'].to_dict()
    C_Ez_Owners = df_Elctro_Costs_index['Owners costs (£/kW)'].to_dict()

    # Fixed replacement costs
    Penalty_Replace_on = {}
    for e in E:
        Penalty_Replace_on[e] = float(df_fixed_rep_costs[e].iloc[0])/(24*365*26*gamma)

    # ---------- CONSTRAINTS ------------

    for t in T:
        prob.addConstraint(P_Grid_b[t] >= 0)
        prob.addConstraint(P_Grid_s[t] >= 0)

    # Power Balance:
    prob.addConstraint( xp.Sum(PPA_sol[r]*scale[r]*P_PPA[(r,y,m,d,h)] for r in R )+ P_Grid_b[(y,m,d,h)]+ P_Bat_out[(y,m,d,h)] == P_Grid_s[(y,m,d,h)] + P_Bat_in[(y,m,d,h)] + xp.Sum(P_Ez[(e,y,m,d,h)] for e in E) for (y,m,d,h) in T)
    
    # Hydrogen Balance:
    if DemandCycle == 'Daily':
        for (y,m,d) in days:
            prob.addConstraint( xp.Sum(H_Ez_out[(e,y,m,d,h)] for e in E for h in hours) ==  D_H2 )
    
    if DemandCycle == 'Weekly':
        unique_weeks = set()
        
        for (y, m, d, h) in T:
            week = day_index[(y, m, d)] // 7
            unique_weeks.add(week)
        
        for week in unique_weeks:
            prob.addConstraint( xp.Sum( H_Ez_out[(e, y, m, d, h)] for e in E for (y, m, d, h) in T if (day_index[(y, m, d)] // 7) == week) == 7 * D_H2 )

    if DemandCycle == 'Monthly':
        unique_months = set((y, m) for (y, m, d, h) in T)
        
        for (y, m) in unique_months:
            days_in_month = len({d for (yy, mm, d, h) in T if yy == y and mm == m})

            prob.addConstraint( xp.Sum( H_Ez_out[(e, yy, mm, dd, hh)] for e in E for (yy, mm, dd, hh) in T if yy == y and mm == m) == days_in_month * D_H2)
    

    # Battery:
    for (y, m, d, h) in T:
        t_next = next_hour(y, m, d, h)
        if t_next in T:
            prob.addConstraint( E_Bat[t_next] == E_Bat[(y, m, d, h)] + Eff_Bat * P_Bat_in[(y, m, d, h)] - P_Bat_out[(y, m, d, h)])
    prob.addConstraint( 0 <= P_Bat_out[t] for t in T )
    prob.addConstraint( 0 <= P_Bat_in[t] for t in T )
    prob.addConstraint( P_Bat_in[t] <= Bat_cap_sol for t in T )
    prob.addConstraint( P_Bat_out[t] <= Bat_cap_sol for t in T )
    prob.addConstraint( 0 <= E_Bat[t] for t in T )
    prob.addConstraint( E_Bat[t]<= Bat_dur*Bat_cap_sol for t in T )

    # prob.addConstraint( P_Bat_in[t] <= 9999999999999*(1-b[t]) for t in T)
    # prob.addConstraint( P_Bat_out[t] <= 9999999999999*b[t] for t in T)
        
    # Average CO2 Emissions:
    if GreenLimit == 'Yearly':
        for y in years:
            prob.addConstraint(xp.Sum( ( Int_Grid[(y_t,m,d,h)]*(P_Grid_b[(y_t,m,d,h)]))/(days_per_year[y_t]*D_H2) for (y_t,m,d,h) in T if y_t == y) <= Int_max)

    if GreenLimit == 'Monthly':
        unique_months = set((y_t, m) for (y_t, m, d, h) in T)
        for (y, m) in unique_months:
            days_in_month = len({d for (yy, mm, d, h) in T if yy == y and mm == m})

            prob.addConstraint(xp.Sum((Int_Grid[(yy, mm, dd, hh)] * P_Grid_b[(yy, mm, dd, hh)]) / (days_in_month * D_H2) for (yy, mm, dd, hh) in T if yy == y and mm == m) <= Int_max)

    if GreenLimit == 'Weekly':
        unique_weeks = set()

        for (y, m, d, h) in T:
            week = day_index[(y, m, d)] // 7
            unique_weeks.add(week)
        
        for week in unique_weeks:
            prob.addConstraint(xp.Sum( ( Int_Grid[(y,m,d,h)]*(P_Grid_b[(y,m,d,h)]))/(7*D_H2) for (y,m,d,h) in T if (day_index[(y, m, d)] // 7) == week) <= Int_max)


    # Electrolysers:
    # prob.addConstraint(Eff_Ez[(e,y,m,d,h)] == ( (x5[e]*Load_Ez[(e,y,m,d,h)])**5 + (x4[e]*Load_Ez[(e,y,m,d,h)])**4 + (x3[e]*Load_Ez[(e,y,m,d,h)])**3 + (x2[e]*Load_Ez[(e,y,m,d,h)])**2 + (x1[e]*Load_Ez[(e,y,m,d,h)]) + x0[e])*x_deg_fact[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)
    prob.addConstraint( P_Ez[(e,y,m,d,h)] == H_Ez_out[(e,y,m,d,h)]*Eff_Ez[e] for e in E for (y,m,d,h) in T)

    # prob.addConstraint( 0 <= P_Ez[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)


    prob.addConstraint( P_Ez[(e,y,m,d,h)] <= z[(e,y,m,d,h)]*Ez_max_load[e]*Ez_cap_sol[e] for e in E for (y,m,d,h) in T)
    prob.addConstraint( z[(e,y,m,d,h)]*Ez_min_load[e]*Ez_cap_sol[e] <= P_Ez[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)

    # prob.addConstraint( Eff_Ez[(e,y,m,d,h)] ==  (mle[e]*Load_Ez[(e,y,m,d,h)] + cle[e])*x_deg_fact[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)
    
    # prob.addConstraint(Ez_cap_sol[e]*Load_Ez[(e,y,m,d,h)] == P_Ez[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)

    prob.addConstraint( z[(e,y,m,d,h)]<= Ez_cap_sol[e] for e in E for (y,m,d,h) in T)
    

    # ---------- OBJECTIVE FUNCTION ----------
    print("LOOK AT THIS DEBUG:",N_periods)
    print("THis is length",len(T))
    print(T)
    # OPEX Costs:
    # Variable OPEX for Power bought and sold on Grid
    variable_OPEX = xp.Sum(C_Grid[(y,m,d,h)]*( P_Grid_b[(y,m,d,h)] - beta*P_Grid_s[(y,m,d,h)]) for (y,m,d,h) in T)
    Av_Hourly_variable_OPEX = variable_OPEX/(N_periods)

    # Stack Replacement Penalty if on
    stack_replacement_penalty = xp.Sum(z[(e,y,m,d,h)]*Penalty_Replace_on[e] for e in E for (y,m,d,h) in T)
    Av_Hourly_replacement_penalty = stack_replacement_penalty/(N_periods)


    prob.setObjective(Av_Hourly_variable_OPEX+Av_Hourly_replacement_penalty, sense = xp.minimize)
    prob.controls.miprelstop = miprelstop
    prob.controls.maxtime = maxtime
    prob.setControl('TIMELIMIT',maxtime)
    prob.solve()

    # ------- PRINT RESULTS ------------

    P_Ez_sol = {(e,y,m,d,h): prob.getSolution(P_Ez[(e,y,m,d,h)]) for (e,y,m,d,h) in P_Ez }
    P_Bat_in_sol = {(y,m,d,h): prob.getSolution(P_Bat_in[(y,m,d,h)]) for (y,m,d,h) in P_Bat_in }
    P_Bat_out_sol = {(y,m,d,h): prob.getSolution(P_Bat_out[(y,m,d,h)]) for (y,m,d,h) in P_Bat_out }
    P_Grid_s_sol = {(y,m,d,h): prob.getSolution(P_Grid_s[(y,m,d,h)]) for (y,m,d,h) in P_Grid_s }
    P_Grid_b_sol = {(y,m,d,h): prob.getSolution(P_Grid_b[(y,m,d,h)])for (y,m,d,h) in P_Grid_b }
    P_PPA_sol = {(r,y,m,d,h): prob.getSolution(scale[r]*P_PPA[(r,y,m,d,h)]*PPA[r]) for (r,y,m,d,h) in P_PPA}
    E_Bat_sol = {(y,m,d,h) : prob.getSolution(E_Bat[(y,m,d,h)]) for (y,m,d,h) in E_Bat }
    ppa = {(r,y,m,d,h): P_PPA[(r,y,m,d,h)] for (r,y,m,d,h) in P_PPA}


    on_hours = prob.getSolution(xp.Sum(z[(e,y,m,d,h)] for (y,m,d,h) in T for e in E))/(10*24)
    # NOTE THIS DOES ASSUME ONLY ONE ELCTROLYSER IS BUILT!!!
    # avg_on = prob.getSolution(xp.Sum(on_hours[e] for e in E))/7*24
    
    # Build base DataFrame with all unique time points
    all_times = set(P_Bat_in_sol.keys()) | set(P_Bat_out_sol.keys()) | set(P_Grid_b_sol.keys()) | set(P_Grid_s_sol.keys())
    all_times |= { (y, m, d, h) for (e, y, m, d, h) in P_Ez_sol }
    all_times |= { (y, m, d, h) for (r, y, m, d, h) in P_PPA_sol }
    all_times |= { (y, m, d, h) for (r, y, m, d, h) in ppa }

    # Convert to datetime index 
    dt_index = pd.to_datetime([f"{y}-{m:02d}-{d:02d} {h-1:02d}:00" for (y, m, d, h) in all_times])
    plot_df = pd.DataFrame(index=dt_index)

    # # Add scalar power flows
    def insert_solution_column(df, sol_dict, column_name):
        s = pd.Series(sol_dict)
        s.index = pd.to_datetime([f"{y}-{m:02d}-{d:02d} {h-1:02d}:00" for (y,m,d,h) in s.index])
        s = s.sort_index()
        df[column_name] = s
    # insert_solution_column(plot_df, P_Grid_b_sol, 'P_Grid_b')
    # insert_solution_column(plot_df, P_Grid_s_sol, 'P_Grid_s')
    # insert_solution_column(plot_df, P_Bat_in_sol, 'P_Bat_in')
    # insert_solution_column(plot_df, P_Bat_out_sol, 'P_Bat_out')
    # insert_solution_column(plot_df, E_Bat_sol, 'E_Bat_sol')

    for e in E:
        # if Ez_cap_sol[e] != 0:
        e_data = { (y, m, d, h): val for (ee, y, m, d, h), val in P_Ez_sol.items() if ee == e }
        s = pd.Series(e_data)
        s.index = pd.to_datetime([f"{y}-{m:02d}-{d:02d} {h-1:02d}:00" for (y, m, d, h) in s.index])
        s = s.sort_index()
        plot_df[f'Power into electrolyser {e}'] = s


    # for r in R:
    #     r_data = { (y, m, d, h): val for (rr, y, m, d, h), val in P_PPA_sol.items() if rr == r }
    #     s = pd.Series(r_data)
    #     s.index = pd.to_datetime([f"{y}-{m:02d}-{d:02d} {h-1:02d}:00" for (y, m, d, h) in s.index])
    #     s = s.sort_index()
    #     plot_df[f'P_PPA_{r}'] = s

    # for r in R:
    #     r_data = { (y, m, d, h): val for (rr, y, m, d, h), val in ppa.items() if rr == r }
    #     s = pd.Series(r_data)
    #     s.index = pd.to_datetime([f"{y}-{m:02d}-{d:02d} {h-1:02d}:00" for (y, m, d, h) in s.index])
    #     s = s.sort_index()
    #     plot_df[f'ppa_{r}'] = s

    # Get the first datetime in the index
    start_time = plot_df.index[0]

    # Extract year and month
    year = start_time.year
    month = start_time.strftime('%B')  


    # # Sort index
    plot_df = plot_df.sort_index()

    # # Filter to a week
    plot_df = plot_df.loc[plot_df.index[0]:plot_df.index[0] + pd.Timedelta(days=7)]

    # # Plot each column on its own subplot
    fig, axs = plt.subplots(len(plot_df.columns), 1, figsize=(14, 2.5 * len(plot_df.columns)), sharex=True)

    for i, col in enumerate(plot_df.columns):
        axs[i].plot(plot_df.index, plot_df[col])
        axs[i].set_ylabel(col)
        axs[i].grid(True)

    fig.suptitle(f"Electrolyser operation during a representative week in {month} {year}", fontsize=14)
    axs[-1].set_xlabel("Time")
    plt.tight_layout()
    # plt.show()

    # plot_df = plot_df.loc[plot_df.index[0]:plot_df.index[0] + pd.Timedelta(days=31)]

    # plt.figure(figsize=(16, 6))

    # for col in plot_df.columns:
    #     plt.plot(plot_df.index, plot_df[col], label=col)

    # plt.title("All Power Flows Over Time")
    # plt.ylabel("Power (kW)")
    # plt.xlabel("Time")
    # plt.legend(loc='upper right', ncol=2)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return on_hours, P_Ez_sol, P_Bat_in_sol, P_Bat_out_sol

# function input set of years & set of months & full df_RE and df_Grid & cap bat & cap ez & ppas & gamma
# find penalty based on gamma
# create a list
# create a counter at 0
# loop over a set of years
# loop over set of months
# df_RE1 = ...from df_RE & grid
# -----solve the optim problem over week: has unit commit, min n max if on, const eff, penalty 
# -----print for each one how it operates label by month name
# add to list the value ehich is number of hrs on /168
# add one to counter for each week
# end both loops find value which is sum of all values in list div by counter
# return the value this becomes new gamma

# do individually gamma func, set new gamma, func , set new gamma, func iterative


   

def find_gamma(year_set,month_set,P_PPA,df_Grid, df_Elctro, df_Elctro_Costs,df_Battery,df_PPA,Demand,DemandCycle,GreenLimit,df_lifetime,df_rep_costs,PPA_sol, Ez_cap_sol, Bat_cap_sol, gamma=1,alpha=1.5,beta=1,Project_Years=26,miprelstop=0.02,maxtime=600):
    '''
    Second part of 2 part model which solves hydrogen production problem for Wood Mackenzie optimising costs and emissions
    Finds proportion of time electrolyser is on on average given representative periods to run unit commitment MINLP problem over
    Runs over the first week for each of the months given in the years given

    Inputs:
        year_set:
        month_set:
        P_PPA: 
        df_Grid: DataFrame with hourly CO2 intensity of power from grid and its price
        df_Elctro: DataFrame with Electrolyser parameters for PEM ALK SOEC
        df_Elctro_Costs: DataFrame with CAPEX costs of electrolysers based on different capacities
        df_Battery: DataFrame with Battery storage costs
        Demand: Demand in kg if daily demand assumed 
        Project_Years: The project lifetime in years ( default at 26 for 2025 - 2050 )
        DemandCycle: The time period where demand must be met, if 'Weekly' then weekly constant demand, if 'Daily' then daily constant demand, if 'Monthly' then monthly constant demand
        GreenLimit: The time period across which the green limit is calculated, daily, weekly, monthly or annual
        alpha: the multiplier for PPA price on calculated 'fairprice' based on grid prices 
        beta: multiplier on selling price of energy back to the grid 
        df_lifetime:
        df_rep_costs:
        gamma:
        PPA_sol:
        Bat_cap_sol:
        Ez_cap_sol:
        targetyear,targetmonth: first week of which month data is saved and presented graphs for 
        miprelstop: MIP gap at which to terminate (default at 2%)
        maxtime: max time to allow the solver to run for (default 120s)
    
    Outputs:
        new_gamma: Estimate for gamma based on unit commitment problem over representative periods given planning decisions
        operational_output.txt file: includes years to replace electrolyser based on operational lifetime ( hours it is on ) and the replacement costs given it is on 100*new_gamma%

    '''
    T = list(
        df_Grid[['Report_Year', 'Report_Month', 'Report_Day', 'Report_Hour']]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    df_Grid_keep =df_Grid
        
    E = list(df_Elctro['Type'].unique())
    R = list(df_PPA['Renewable Source'].unique())
        
    days = sorted(set((y, m, d) for (y, m, d, h) in T))
    years = sorted(set(y for (y,m,d,h) in T))   
    months = range(1,13)
    hours = range(1,25)

    with open("operational_output.txt", "w") as f:
        proportion_of_hours_on_each_week = []
        counter = 0

        for y in year_set:
            for m in month_set:
                week_P_PPA = {}
                for r in R:
                    for (yy, mm, d, h) in T:
                        if yy == y and mm == m and d <= 10:
                            week_P_PPA[r, y, m, d, h] = P_PPA[r, y, m, d, h]
                week_df_Grid = {}
                week_df_Grid = df_Grid_keep[(df_Grid_keep["Report_Year"] == y) & (df_Grid_keep["Report_Month"] == m )& (df_Grid_keep["Report_Day"] <= 10)]
                print(week_df_Grid)
                avg_hours_on,x,xx,xxx = operational_model(P_PPA=week_P_PPA,df_Grid=week_df_Grid, df_Elctro=df_Elctro, df_Elctro_Costs=df_Elctro_Costs,df_Battery=df_Battery,df_PPA=df_PPA,Demand=Demand,DemandCycle=DemandCycle,GreenLimit=GreenLimit,df_lifetime=df_lifetime,df_rep_costs=df_rep_costs,PPA_sol=PPA_sol, Ez_cap_sol=Ez_cap_sol, Bat_cap_sol=Bat_cap_sol, gamma=gamma,alpha=alpha,beta=beta,Project_Years=Project_Years,miprelstop=miprelstop,maxtime=maxtime)
                
                print(f'Representative Week {counter+1}: Month {m}, Year {y}, Average proportion of time the electrolyser is on is {100*avg_hours_on}%', file = f)
                proportion_of_hours_on_each_week.append(avg_hours_on)
                # print(proportion_of_hours_on_each_week)
                counter += 1
                # print(counter)
        
        new_gamma = sum(proportion_of_hours_on_each_week)/counter

        for e in E:
            if Ez_cap_sol[e] != 0:
                print(f'Final average proportion of time {e} electrolyser is on is {new_gamma*100}%',file=f)

                df_rep_years_new_gamma, df_fixed_rep_costs_new_gamma = find_fixed_replace(new_gamma,df_Elctro,df_lifetime,df_rep_costs)
            
            
                print(f'With the assumption that the {e} electrolyser is on {new_gamma*100}% of time, the electrolyser stack is replaced in the following years: ',file=f)
                for i in range(len(df_rep_years_new_gamma[e])):
                    print(f"Replace in year {df_rep_years_new_gamma[e][i]}",file=f)
                # print(f'The total replacement costs for electrolyser {e}  over the time horizon will be £{float(df_fixed_rep_costs_new_gamma[e].iloc[0])}',file=f)
                
            

    return new_gamma, proportion_of_hours_on_each_week,x,xx,xxx
    
def iteratively_find_best_gamma(year_set,month_set,df_RE,df_Grid, df_Elctro, df_Elctro_Costs,df_Battery,df_PPA,Demand,DemandCycle,GreenLimit,df_lifetime,df_rep_costs,PPA_sol, Ez_cap_sol, Bat_cap_sol, gamma=1,alpha=1.5,beta=1,Project_Years=26,miprelstop=0.02,maxtime=600):
    '''
    Iterating second part of 2 part model which solves hydrogen production problem for Wood Mackenzie optimising costs and emissions
    Finds proportion of time electrolyser is on on average given representative periods to run unit commitment MINLP problem over
    Runs over the first week for each of the months given in the years given
    It runs the above iteratively until the initial gamma and the new gamma are close in values

    Inputs:
        year_set:
        month_set:
        df_RE: DataFrame with hourly solar, offshore wind, onshore wind data
        df_Grid: DataFrame with hourly CO2 intensity of power from grid and its price
        df_Elctro: DataFrame with Electrolyser parameters for PEM ALK SOEC
        df_Elctro_Costs: DataFrame with CAPEX costs of electrolysers based on different capacities
        df_Battery: DataFrame with Battery storage costs
        Demand: Demand in kg if daily demand assumed 
        Project_Years: The project lifetime in years ( default at 26 for 2025 - 2050 )
        DemandCycle: The time period where demand must be met, if 'Weekly' then weekly constant demand, if 'Daily' then daily constant demand, if 'Monthly' then monthly constant demand
        GreenLimit: The time period across which the green limit is calculated, daily, weekly, monthly or annual
        alpha: the multiplier for PPA price on calculated 'fairprice' based on grid prices 
        beta: multiplier on selling price of energy back to the grid 
        df_lifetime:
        df_rep_costs:
        gamma:
        PPA_sol:
        Bat_cap_sol:
        Ez_cap_sol:
        targetyear,targetmonth: first week of which month data is saved and presented graphs for 
        miprelstop: MIP gap at which to terminate (default at 2%)
        maxtime: max time to allow the solver to run for (default 120s)
    
    Outputs:
        Estimate for gamma based on unit commitment problem over representative periods given planning decisions


    '''

    old_gamma = 5
    new_gamma = gamma

    tolerance = 0.001

    while abs(new_gamma - old_gamma) > tolerance:
        old_gamma = new_gamma
        new_gamma = find_gamma(year_set,month_set,df_RE,df_Grid, df_Elctro, df_Elctro_Costs,df_Battery,df_PPA,Demand,DemandCycle,GreenLimit,df_lifetime,df_rep_costs,PPA_sol, Ez_cap_sol, Bat_cap_sol, old_gamma,alpha,beta,Project_Years,miprelstop,maxtime)
        
    df_rep_years_new_gamma, df_fixed_rep_costs_new_gamma = find_fixed_replace(new_gamma,df_Elctro,df_lifetime,df_rep_costs)
    
    return new_gamma, df_rep_years_new_gamma, df_fixed_rep_costs_new_gamma

    

def operational_model_eff(P_PPA,df_Grid, df_Elctro, df_Elctro_Costs,df_Battery,df_PPA,Demand,DemandCycle,GreenLimit,df_lifetime,df_rep_costs,PPA_sol, Ez_cap_sol, Bat_cap_sol, gamma=1,alpha=1.5,beta=1,Project_Years=26,miprelstop=0.02,maxtime=600):
    '''
    The model which solves hydrogen production problem for Wood Mackenzie optimising costs and emissions
    It includes unit commitment and finds when the elctrolyser is on and off, hence includes min/max load and replacment cost as a penalty
    MINLP which has a penalty for electrolyser being on, calculated as a fixed cost using df_lifetime and df_rep_costs assuming the elctrolyser was on gamma*100% of the time
    Assumes constant efficiency of electrolysers
    Solves assuming the data inputted is only for a week

    Inputs:
        P_PPA:
        df_Grid: DataFrame with hourly CO2 intensity of power from grid and its price for a week of dat
        df_Elctro: DataFrame with Electrolyser parameters for PEM ALK SOEC
        df_Elctro_Costs: DataFrame with CAPEX costs of electrolysers based on different capacities
        df_Battery: DataFrame with Battery storage costs
        Demand: Demand in kg if daily demand assumed 
        Project_Years: The project lifetime in years ( default at 26 for 2025 - 2050 )
        DemandCycle: The time period where demand must be met, if 'Weekly' then weekly constant demand, if 'Daily' then daily constant demand, if 'Monthly' then monthly constant demand
        GreenLimit: The time period across which the green limit is calculated, daily, weekly, monthly or annual
        alpha: the multiplier for PPA price on calculated 'fairprice' based on grid prices 
        beta: multiplier on selling price of energy back to the grid 
        df_lifetime:
        df_rep_costs:
        gamma:
        PPA_sol:
        Bat_cap_sol:
        Ez_cap_sol:
        targetyear,targetmonth: first week of which month data is saved and presented graphs for 
        miprelstop: MIP gap at which to terminate (default at 2%)
        maxtime: max time to allow the solver to run for (default 120s)
    
    Outputs:
        Plots operation over the time horizon the model is run for
        Gives the number of hours the electrolyser is on for over the time horizon


    ''' 
    xp.init('/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr')

    prob = xp.problem(name="Hydrogen WoodMac")

    def clean_name(r):
        return r.replace(' ', '_')



    # ------------ SETS -------------
    T = list(
        df_Grid[['Report_Year', 'Report_Month', 'Report_Day', 'Report_Hour']]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
        
    E = list(df_Elctro['Type'].unique())
    R = list(df_PPA['Renewable Source'].unique())
        
    days = sorted(set((y, m, d) for (y, m, d, h) in T))
    years = sorted(set(y for (y,m,d,h) in T))   
    months = range(1,13)
    hours = range(1,25)

    df_T = pd.DataFrame(T, columns=['y', 'm', 'd', 'h'])
    days_per_year = df_T.drop_duplicates(['y', 'm', 'd']).groupby('y').size().to_dict()

    def next_hour(y, m, d, h):
        # Convert hour from 1–24 to 0–23
        dt = datetime(y, m, d, h - 1)
        dt_next = dt + timedelta(hours=1)
            # Convert back to 1–24 format
        return (dt_next.year, dt_next.month, dt_next.day, dt_next.hour + 1)

    def prev_hour(y, m, d, h):
        dt = datetime(y, m, d, h - 1)
        dt_prev = dt - timedelta(hours=1)
        dt_prev_hour = dt_prev.hour +1
        return dt_prev.year, dt_prev.month, dt_prev.day, dt_prev_hour
        
    # Index Grid data by (year, month, date, hour)
    df_Grid_index = df_Grid.copy()
    df_Grid_index.set_index(['Report_Year', 'Report_Month', 'Report_Day', 'Report_Hour'],inplace = True)
    df_PPA_index = df_PPA.copy()
    df_PPA_index.set_index('Renewable Source', inplace=True)
    df_Elctro_Costs_index = df_Elctro_Costs.copy()
    df_Elctro_Costs_index.set_index('Technology', inplace=True)

    df_rep_years, df_fixed_rep_costs = find_fixed_replace(gamma,df_Elctro,df_lifetime,df_rep_costs)

    # ------ DECISION VARIABLES ------

    # Proportion of renewable energy contracted to take from renewable site r
    PPA = {(r): xp.var(vartype=xp.continuous, name = f'PPA_{clean_name(r)}', lb= 0, ub=1) for r in R}

    # Power bought from the grid at time t (kW)
    P_Grid_b = {(y,m,d,h): xp.var(vartype=xp.continuous, name = f'P_Grid_b_{y}_{m}_{d}_{h}', lb=0) for (y,m,d,h) in T}

    # Power sold to the grid at time t (kW)
    P_Grid_s = {(y,m,d,h): xp.var(vartype=xp.continuous, name = f'P_Grid_s_{y}_{m}_{d}_{h}',lb=0) for (y,m,d,h) in T}

    # Power taken out of battery at time t (kW)
    P_Bat_out = {(y,m,d,h): xp.var(vartype=xp.continuous, name = f'P_Bat_out_{y}_{m}_{d}_{h}', lb=0) for (y,m,d,h) in T}

    # Power put into battery at time t (kW)
    P_Bat_in = {(y,m,d,h): xp.var(vartype=xp.continuous, name = f'P_Bat_in_{y}_{m}_{d}_{h}',lb=0) for (y,m,d,h) in T}

    # Power put into electrolyser e at time t (kW)
    P_Ez = {(e,y,m,d,h): xp.var(vartype=xp.continuous, name = f'P_Ez_{e}_{y}_{m}_{d}_{h}',lb=0) for e in E for (y,m,d,h) in T}

    # # Power required for putting H2 into storage at time t (kW)
    # P_H2st = {(t): xp.var(vartype=xp.continuous, name = f'P_H2st_{y}_{m}_{d}_{h}') for t in T}

    # # Hydrogen leaving store at time t (kg/h)
    # H_H2st_out = {(t): xp.var(vartype=xp.continuous, name = f'P_H2st_out_{y}_{m}_{d}_{h}') for t in T}

    # # Hydrogen entering store at time t (kg/h)
    # H_H2st_in = {(t): xp.var(vartype=xp.continuous, name = f'P_H2st_in_{t}') for t in T}

    #  Hydrogen leaving electrolyser e at time t (kg/h)
    H_Ez_out = {(e, y,m,d,h): xp.var(vartype=xp.continuous, name = f'H_Ez_out_{e}_{y}_{m}_{d}_{h}',lb=0) for e in E for (y,m,d,h) in T}

    # Energy stored in battery at time t (kWh)
    E_Bat = {(y,m,d,h): xp.var(vartype=xp.continuous, name = f'E_Bat_{y}_{m}_{d}_{h}',lb=0) for (y,m,d,h) in T}

    # # Hydrogen stored at time t (kg)
    # E_H2st = {(t): xp.var(vartype=xp.continuous, name = f'E_H2st_{t}') for t in T}

    # # Energy capacity of H2 storage tank (kg)
    # Q_H2st_cap = xp.var(vartype=xp.continuous, name='Q_H2st_cap')

    # Load factor of electrolyser e at time t
    Load_Ez = {(e, y,m,d,h): xp.var(vartype=xp.continuous, name = f'Load_Ez_{e}_{y}_{m}_{d}_{h}',lb=0,ub=1) for e in E for (y,m,d,h) in T}

    # Binary variable for if electrolyser e is on at time t
    z = {(e, y,m,d,h): xp.var(vartype=xp.binary, name = f'z_{e}_{y}_{m}_{d}_{h}') for e in E for (y,m,d,h) in T}

    # Energy needed for electrolyser e per kg of hydrogen output as  a function of the load factor (kWh/kg)
    Eff_Ez = {(e,y,m,d,h): xp.var(vartype=xp.continuous, name = f'Eff_Ez{e}_{y}_{m}_{d}_{h}',lb=0) for e in E for (y,m,d,h) in T}

    # If electrolyser e is built or not
    # build = {(e): xp.var(vartype=xp.binary, name = f'build_{e}') for e in E }

    prob.addVariable(PPA, P_Grid_b, P_Grid_s, P_Bat_out, P_Bat_in, P_Ez, H_Ez_out, E_Bat,z,Load_Ez,Eff_Ez)

    # --------- INDEX DATA ------------

    # Index electrolyser data by electrolyser name
    df_Elctro_index = df_Elctro.copy()
    df_Elctro_index.set_index('Type', inplace=True)

    # Find day indexes
    day_index = {}
    counter = 0
    for (y, m, d, h) in sorted(T):  
        if (y, m, d) not in day_index:
            day_index[(y, m, d)] = counter
            counter += 1

    # --------- PARAMETERS ------------

    #  Power available from renewable site r at time t (kW) = P_PPA

    # Capacity size as a scale for each renewable r (kW) used if given data needed to be normalised and scaled for capacity
    scale = df_PPA_index['Renewable Capacity Scale (kW)'].to_dict()

    # Constant on-site daily hydrogen demand (kg)
    D_H2 = float(Demand)

    # Round-trip efficiency for battery
    Eff_Bat = float(df_Battery["Round trip efficiency"].iloc[0])

    # Electrolyser Simple efficiency
    # Eff_Ez = df_Elctro_index['Simple Efficiency'].to_dict()

    # CO2 Intensity of power from the grid at time t (kg/kWh)
    Int_Grid = df_Grid_index['CO2 Intensity (kg CO2/kWh)'].to_dict()

    # Maximum CO2 emissions in kg CO2 e/kg H2
    Int_max = 2.4

    # Min and Max load for electrolyser e
    Ez_min_load = df_Elctro_index['Minimum Load'].to_dict()
    Ez_max_load = df_Elctro_index['Maximum Load'].to_dict()

    # Duration in hours of the battery
    Bat_dur = float(df_Battery['Duration (hrs)'].iloc[0])

    # Number of years over time horizon
    N_years = len(years)
    N_periods = len(T)

    # Final Investment Decison Year
    x_FID = 2025

    # System efficiency degredation 
    x_eff_deg = df_Elctro_index['System Efficiency Degradation'].to_dict()

    # System degredation factor 
    x_deg_fact = {}
    for e in E:
        for (y,m,d,h) in T:
            y_int = int(y)
            x_deg_fact[(e,y,m,d,h)] = x_eff_deg[e]**(x_FID - y_int)

    # Coefficients for linear efficiency
    mle = df_Elctro_index['m'].to_dict()
    cle = df_Elctro_index['c'].to_dict()

    # Cost for buying and selling power from the grid
    C_Grid = df_Grid_index['Price (£/kWh, real 2025)'].to_dict()

    # CAPEX Cost for capacity of battery store
    C_Bat_capex = float(df_Battery['Capex (£/kWh)'].iloc[0])

    # CAPEX Cost for electrolyser e
    C_Ez_capex = df_Elctro_Costs_index['Total Installed Cost (TIC) (£/kW)'].to_dict()

    # Fixed OPEX cost for electrolyser e as proportion of capex
    C_Ez_fixed_opex = df_Elctro_index['Fixed Opex percent'].to_dict()

    # Fixed OPEX cost for battery as proportion of capex
    C_Bat_fixed_opex = float(df_Battery['Fixed Opex percent'].iloc[0])

    # Costs which make up CAPEX cost of electrolysers
    C_Ez_BoS = df_Elctro_Costs_index['Balance of Stack (£/kW)'].to_dict()
    C_Ez_BoP = df_Elctro_Costs_index['Balance of Plant (£/kW)'].to_dict()
    C_Ez_EPC = df_Elctro_Costs_index['Engineering, Procurement & Construction costs (£/kW)'].to_dict()
    C_Ez_Owners = df_Elctro_Costs_index['Owners costs (£/kW)'].to_dict()

    # Fixed replacement costs
    Penalty_Replace_on = {}
    for e in E:
        Penalty_Replace_on[e] = float(df_fixed_rep_costs[e].iloc[0])/(24*365*26*gamma)

    # ---------- CONSTRAINTS ------------

    for t in T:
        prob.addConstraint(P_Grid_b[t] >= 0)
        prob.addConstraint(P_Grid_s[t] >= 0)

    # Power Balance:
    prob.addConstraint( xp.Sum(PPA_sol[r]*scale[r]*P_PPA[(r,y,m,d,h)] for r in R )+ P_Grid_b[(y,m,d,h)]+ P_Bat_out[(y,m,d,h)] == P_Grid_s[(y,m,d,h)] + P_Bat_in[(y,m,d,h)] + xp.Sum(P_Ez[(e,y,m,d,h)] for e in E) for (y,m,d,h) in T)
    
    # Hydrogen Balance:
    if DemandCycle == 'Daily':
        for (y,m,d) in days:
            prob.addConstraint( xp.Sum(H_Ez_out[(e,y,m,d,h)] for e in E for h in hours) ==  D_H2 )
    
    if DemandCycle == 'Weekly':
        unique_weeks = set()
        
        for (y, m, d, h) in T:
            week = day_index[(y, m, d)] // 7
            unique_weeks.add(week)
        
        for week in unique_weeks:
            prob.addConstraint( xp.Sum( H_Ez_out[(e, y, m, d, h)] for e in E for (y, m, d, h) in T if (day_index[(y, m, d)] // 7) == week) == 7 * D_H2 )

    if DemandCycle == 'Monthly':
        unique_months = set((y, m) for (y, m, d, h) in T)
        
        for (y, m) in unique_months:
            days_in_month = len({d for (yy, mm, d, h) in T if yy == y and mm == m})

            prob.addConstraint( xp.Sum( H_Ez_out[(e, yy, mm, dd, hh)] for e in E for (yy, mm, dd, hh) in T if yy == y and mm == m) == days_in_month * D_H2)
    

    # Battery:
    for (y, m, d, h) in T:
        t_next = next_hour(y, m, d, h)
        if t_next in T:
            prob.addConstraint( E_Bat[t_next] == E_Bat[(y, m, d, h)] + Eff_Bat*P_Bat_in[(y, m, d, h)] - P_Bat_out[(y, m, d, h)])
    prob.addConstraint( 0 <= P_Bat_out[t] for t in T )
    prob.addConstraint( 0 <= P_Bat_in[t] for t in T )
    prob.addConstraint( P_Bat_in[t] <= Bat_cap_sol for t in T )
    prob.addConstraint( P_Bat_out[t] <= Bat_cap_sol for t in T )
    prob.addConstraint( 0 <= E_Bat[t] for t in T )
    prob.addConstraint( E_Bat[t]<= Bat_dur*Bat_cap_sol for t in T )

    # prob.addConstraint( P_Bat_in[t] <= 9999999999999*(1-b[t]) for t in T)
    # prob.addConstraint( P_Bat_out[t] <= 9999999999999*b[t] for t in T)
        
    # Average CO2 Emissions:
    if GreenLimit == 'Yearly':
        for y in years:
            prob.addConstraint(xp.Sum( ( Int_Grid[(y_t,m,d,h)]*(P_Grid_b[(y_t,m,d,h)]))/(days_per_year[y_t]*D_H2) for (y_t,m,d,h) in T if y_t == y) <= Int_max)

    if GreenLimit == 'Monthly':
        unique_months = set((y_t, m) for (y_t, m, d, h) in T)
        for (y, m) in unique_months:
            days_in_month = len({d for (yy, mm, d, h) in T if yy == y and mm == m})

            prob.addConstraint(xp.Sum((Int_Grid[(yy, mm, dd, hh)] * P_Grid_b[(yy, mm, dd, hh)]) / (days_in_month * D_H2) for (yy, mm, dd, hh) in T if yy == y and mm == m) <= Int_max)

    if GreenLimit == 'Weekly':
        unique_weeks = set()

        for (y, m, d, h) in T:
            week = day_index[(y, m, d)] // 7
            unique_weeks.add(week)
        
        for week in unique_weeks:
            prob.addConstraint(xp.Sum( ( Int_Grid[(y,m,d,h)]*(P_Grid_b[(y,m,d,h)]))/(7*D_H2) for (y,m,d,h) in T if (day_index[(y, m, d)] // 7) == week) <= Int_max)


    # Electrolysers:
    # prob.addConstraint(Eff_Ez[(e,y,m,d,h)] == ( (x5[e]*Load_Ez[(e,y,m,d,h)])**5 + (x4[e]*Load_Ez[(e,y,m,d,h)])**4 + (x3[e]*Load_Ez[(e,y,m,d,h)])**3 + (x2[e]*Load_Ez[(e,y,m,d,h)])**2 + (x1[e]*Load_Ez[(e,y,m,d,h)]) + x0[e])*x_deg_fact[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)
    prob.addConstraint( P_Ez[(e,y,m,d,h)] == H_Ez_out[(e,y,m,d,h)]*Eff_Ez[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)

    # prob.addConstraint( 0 <= P_Ez[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)


    prob.addConstraint( P_Ez[(e,y,m,d,h)] <= z[(e,y,m,d,h)]*Ez_max_load[e]*Ez_cap_sol[e] for e in E for (y,m,d,h) in T)
    prob.addConstraint( z[(e,y,m,d,h)]*Ez_min_load[e]*Ez_cap_sol[e] <= P_Ez[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)

    prob.addConstraint( Eff_Ez[(e,y,m,d,h)] ==  (mle[e]*Load_Ez[(e,y,m,d,h)] + cle[e])*x_deg_fact[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)
    
    prob.addConstraint(Ez_cap_sol[e]*Load_Ez[(e,y,m,d,h)] == P_Ez[(e,y,m,d,h)] for e in E for (y,m,d,h) in T)

    prob.addConstraint( z[(e,y,m,d,h)]<= Ez_cap_sol[e] for e in E for (y,m,d,h) in T)
    

    # ---------- OBJECTIVE FUNCTION ----------

    # OPEX Costs:
    # Variable OPEX for Power bought and sold on Grid
    variable_OPEX = xp.Sum(C_Grid[(y,m,d,h)]*( P_Grid_b[(y,m,d,h)] - beta*P_Grid_s[(y,m,d,h)]) for (y,m,d,h) in T)
    Av_Hourly_variable_OPEX = variable_OPEX/N_periods

    # Stack Replacement Penalty if on
    stack_replacement_penalty = xp.Sum(z[(e,y,m,d,h)]*Penalty_Replace_on[e] for e in E for (y,m,d,h) in T)
    Av_Hourly_replacement_penalty = stack_replacement_penalty/N_periods


    prob.setObjective(Av_Hourly_variable_OPEX+Av_Hourly_replacement_penalty, sense = xp.minimize)
    prob.controls.miprelstop = miprelstop
    prob.controls.maxtime = maxtime
    prob.setControl('TIMELIMIT',maxtime)
    prob.solve()

    # ------- PRINT RESULTS ------------

    P_Ez_sol = {(e,y,m,d,h): prob.getSolution(P_Ez[(e,y,m,d,h)]) for (e,y,m,d,h) in P_Ez }
    P_Bat_in_sol = {(y,m,d,h): prob.getSolution(P_Bat_in[(y,m,d,h)]) for (y,m,d,h) in P_Bat_in }
    P_Bat_out_sol = {(y,m,d,h): prob.getSolution(P_Bat_out[(y,m,d,h)]) for (y,m,d,h) in P_Bat_out }
    P_Grid_s_sol = {(y,m,d,h): prob.getSolution(P_Grid_s[(y,m,d,h)]) for (y,m,d,h) in P_Grid_s }
    P_Grid_b_sol = {(y,m,d,h): prob.getSolution(P_Grid_b[(y,m,d,h)])for (y,m,d,h) in P_Grid_b }
    P_PPA_sol = {(r,y,m,d,h): prob.getSolution(scale[r]*P_PPA[(r,y,m,d,h)]*PPA[r]) for (r,y,m,d,h) in P_PPA}
    E_Bat_sol = {(y,m,d,h) : prob.getSolution(E_Bat[(y,m,d,h)]) for (y,m,d,h) in E_Bat }
    ppa = {(r,y,m,d,h): P_PPA[(r,y,m,d,h)] for (r,y,m,d,h) in P_PPA}


    on_hours = prob.getSolution(xp.Sum(z[(e,y,m,d,h)] for (y,m,d,h) in T for e in E))/(10*24)
    # NOTE THIS DOES ASSUME ONLY ONE ELCTROLYSER IS BUILT!!!
    # avg_on = prob.getSolution(xp.Sum(on_hours[e] for e in E))/7*24
    
    # Build base DataFrame with all unique time points
    all_times = set(P_Bat_in_sol.keys()) | set(P_Bat_out_sol.keys()) | set(P_Grid_b_sol.keys()) | set(P_Grid_s_sol.keys())
    all_times |= { (y, m, d, h) for (e, y, m, d, h) in P_Ez_sol }
    all_times |= { (y, m, d, h) for (r, y, m, d, h) in P_PPA_sol }
    all_times |= { (y, m, d, h) for (r, y, m, d, h) in ppa }

    # Convert to datetime index 
    dt_index = pd.to_datetime([f"{y}-{m:02d}-{d:02d} {h-1:02d}:00" for (y, m, d, h) in all_times])
    plot_df = pd.DataFrame(index=dt_index)

    # # Add scalar power flows
    def insert_solution_column(df, sol_dict, column_name):
        s = pd.Series(sol_dict)
        s.index = pd.to_datetime([f"{y}-{m:02d}-{d:02d} {h-1:02d}:00" for (y,m,d,h) in s.index])
        s = s.sort_index()
        df[column_name] = s
    # insert_solution_column(plot_df, P_Grid_b_sol, 'P_Grid_b')
    # insert_solution_column(plot_df, P_Grid_s_sol, 'P_Grid_s')
    # insert_solution_column(plot_df, P_Bat_in_sol, 'P_Bat_in')
    # insert_solution_column(plot_df, P_Bat_out_sol, 'P_Bat_out')
    # insert_solution_column(plot_df, E_Bat_sol, 'E_Bat_sol')

    for e in E:
        # if Ez_cap_sol[e] != 0:
        e_data = { (y, m, d, h): val for (ee, y, m, d, h), val in P_Ez_sol.items() if ee == e }
        s = pd.Series(e_data)
        s.index = pd.to_datetime([f"{y}-{m:02d}-{d:02d} {h-1:02d}:00" for (y, m, d, h) in s.index])
        s = s.sort_index()
        plot_df[f'Power into electrolyser {e}'] = s


    # for r in R:
    #     r_data = { (y, m, d, h): val for (rr, y, m, d, h), val in P_PPA_sol.items() if rr == r }
    #     s = pd.Series(r_data)
    #     s.index = pd.to_datetime([f"{y}-{m:02d}-{d:02d} {h-1:02d}:00" for (y, m, d, h) in s.index])
    #     s = s.sort_index()
    #     plot_df[f'P_PPA_{r}'] = s

    # for r in R:
    #     r_data = { (y, m, d, h): val for (rr, y, m, d, h), val in ppa.items() if rr == r }
    #     s = pd.Series(r_data)
    #     s.index = pd.to_datetime([f"{y}-{m:02d}-{d:02d} {h-1:02d}:00" for (y, m, d, h) in s.index])
    #     s = s.sort_index()
    #     plot_df[f'ppa_{r}'] = s

    # Get the first datetime in the index
    start_time = plot_df.index[0]

    # Extract year and month
    year = start_time.year
    month = start_time.strftime('%B')  


    # # Sort index
    plot_df = plot_df.sort_index()

    # # Filter to a week
    plot_df = plot_df.loc[plot_df.index[0]:plot_df.index[0] + pd.Timedelta(days=7)]

    # # Plot each column on its own subplot
    fig, axs = plt.subplots(len(plot_df.columns), 1, figsize=(14, 2.5 * len(plot_df.columns)), sharex=True)

    for i, col in enumerate(plot_df.columns):
        axs[i].plot(plot_df.index, plot_df[col])
        axs[i].set_ylabel(col)
        axs[i].grid(True)

    fig.suptitle(f"Electrolyser operation during a representative week in {month} {year}", fontsize=14)
    axs[-1].set_xlabel("Time")
    plt.tight_layout()
    # plt.show()

    # plot_df = plot_df.loc[plot_df.index[0]:plot_df.index[0] + pd.Timedelta(days=31)]

    # plt.figure(figsize=(16, 6))

    # for col in plot_df.columns:
    #     plt.plot(plot_df.index, plot_df[col], label=col)

    # plt.title("All Power Flows Over Time")
    # plt.ylabel("Power (kW)")
    # plt.xlabel("Time")
    # plt.legend(loc='upper right', ncol=2)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return on_hours, P_Ez_sol, P_Bat_in_sol, P_Bat_out_sol

