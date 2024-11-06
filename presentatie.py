import pandas as pd
import numpy as np
import seaborn as sns
import folium
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
import os
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from vinc import v_direct 
import geopandas as gpd
from statistics import mean
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title='Vliegtuigen informatie',
    page_icon="💺",
    layout = "wide",
    initial_sidebar_state = "expanded")

st.title("Analyse vliegtuigen")

# Datasets inlezen
@st.cache_data # Zorgt ervoor dat de functie in dit geval maar 1 keer wordt uitgevoerd
def get_data(): # Haalt alle data op

    df = pd.read_csv('schedule_airport.csv', sep = ",")  # Leest het CSV-bestand

    df['STA_STD_ltc'] = pd.to_datetime(df['STA_STD_ltc'], format = '%H:%M:%S' , errors = 'coerce')
    df['ATA_ATD_ltc'] = pd.to_datetime(df['ATA_ATD_ltc'], format = '%H:%M:%S' , errors = 'coerce')

    # Data aan maken voor statistische analyse
    df['Delay_Minutes'] = (df['ATA_ATD_ltc'] - df['STA_STD_ltc']).dt.total_seconds() / 60
    df_cleaned = df.dropna(subset = ['Delay_Minutes'])
    df_cleaned['STD'] = pd.to_datetime(df_cleaned['STD'], format = '%d/%m/%Y', errors = 'coerce')   
    daily_delay = df_cleaned.groupby(df_cleaned['STD'].dt.date)['Delay_Minutes'].mean()

    

    df_airports = pd.read_csv("airports-extended-clean.csv", sep = ";")
    df_airports = df_airports[(df_airports['Type'] == 'airport') & (df_airports['Source'] == 'OurAirports')]
    df_schedule = pd.read_csv("schedule_airport.csv", sep = ",")

    df_airports['Longitude'] = df_airports['Longitude'].astype(str).str.replace(',', '.')
    df_airports['Latitude'] = df_airports['Latitude'].astype(str).str.replace(',', '.')
    df_airports['Longitude'] = pd.to_numeric(df_airports['Longitude'])
    df_airports['Latitude'] = pd.to_numeric(df_airports['Latitude'])

    df_airports_schedule = df_airports.merge(df_schedule, left_on="ICAO", right_on="Org/Des")
    df_airports_schedule.dropna(inplace = True)

    df_airports_schedule["Latitude"] = df_airports_schedule["Latitude"].replace(',','.',regex=True)
    df_airports_schedule["Longitude"] = df_airports_schedule["Longitude"].replace(',','.',regex=True)
    df_airports_schedule["Timezone"] = df_airports_schedule["Timezone"].replace(',','.',regex=True)
    

    df_airports_schedule["STA_STD_ltc"] = pd.to_datetime(df_airports_schedule["STD"] + " " + df_airports_schedule["STA_STD_ltc"], format="%d/%m/%Y %H:%M:%S")
    df_airports_schedule["ATA_ATD_ltc"] = pd.to_datetime(df_airports_schedule["STD"] + " " + df_airports_schedule["ATA_ATD_ltc"], format="%d/%m/%Y %H:%M:%S")
    df_airports_schedule["STD"] = pd.to_datetime(df_airports_schedule["STD"], format="%d/%m/%Y")
    df_airports_schedule = df_airports_schedule.astype({'Latitude': float, 'Longitude': float, "Timezone": float})

    df_airports_schedule["Delay"] = (df_airports_schedule["ATA_ATD_ltc"] - df_airports_schedule["STA_STD_ltc"]) / pd.Timedelta(seconds=1)

    # Vluchten per seconde inlezen
    flight1 = pd.read_excel('1Flight 1.xlsx')
    flight2 = pd.read_excel('1Flight 2.xlsx')
    flight3 = pd.read_excel('1Flight 3.xlsx')
    flight4 = pd.read_excel('1Flight 4.xlsx')
    flight5 = pd.read_excel('1Flight 5.xlsx')
    flight6 = pd.read_excel('1Flight 6.xlsx')
    flight7 = pd.read_excel('1Flight 7.xlsx')

    # Vluchten per 30 seconde inlezen 
    flight1_30sec = pd.read_excel('30Flight 1.xlsx', names=['Time', 'Latitude', 'Longitude', 'AltitudeM', 'Altitudeft', 'Heading', 'TRUE AIRSPEED (derived)'])
    flight2_30sec = pd.read_excel('30Flight 2.xlsx', names=['Time', 'Latitude', 'Longitude', 'AltitudeM', 'Altitudeft', 'Heading', 'TRUE AIRSPEED (derived)'])
    flight3_30sec = pd.read_excel('30Flight 3.xlsx', names=['Time', 'Latitude', 'Longitude', 'AltitudeM', 'Altitudeft', 'Heading', 'TRUE AIRSPEED (derived)'])
    flight4_30sec = pd.read_excel('30Flight 4.xlsx', names=['Time', 'Latitude', 'Longitude', 'AltitudeM', 'Altitudeft', 'Heading', 'TRUE AIRSPEED (derived)'])
    flight5_30sec = pd.read_excel('30Flight 5.xlsx', names=['Time', 'Latitude', 'Longitude', 'AltitudeM', 'Altitudeft', 'Heading', 'TRUE AIRSPEED (derived)'])
    flight6_30sec = pd.read_excel('30Flight 6.xlsx', names=['Time', 'Latitude', 'Longitude', 'AltitudeM', 'Altitudeft', 'Heading', 'TRUE AIRSPEED (derived)'])
    flight7_30sec = pd.read_excel('30Flight 7.xlsx', names=['Time', 'Latitude', 'Longitude', 'AltitudeM', 'Altitudeft', 'Heading', 'TRUE AIRSPEED (derived)'])

    # Types veranderen van True Airspeed van object naar float 
    flight1['TRUE AIRSPEED (derived)'] = pd.to_numeric(flight1['TRUE AIRSPEED (derived)'], errors='coerce')
    flight2['TRUE AIRSPEED (derived)'] = pd.to_numeric(flight2['TRUE AIRSPEED (derived)'], errors='coerce')
    flight3['TRUE AIRSPEED (derived)'] = pd.to_numeric(flight3['TRUE AIRSPEED (derived)'], errors='coerce')
    flight4['TRUE AIRSPEED (derived)'] = pd.to_numeric(flight4['TRUE AIRSPEED (derived)'], errors='coerce')
    flight5['TRUE AIRSPEED (derived)'] = pd.to_numeric(flight5['TRUE AIRSPEED (derived)'], errors='coerce')
    flight6['TRUE AIRSPEED (derived)'] = pd.to_numeric(flight6['TRUE AIRSPEED (derived)'], errors='coerce')
    flight7['TRUE AIRSPEED (derived)'] = pd.to_numeric(flight7['TRUE AIRSPEED (derived)'], errors='coerce')

    flight1_30sec['TRUE AIRSPEED (derived)'] = pd.to_numeric(flight1_30sec['TRUE AIRSPEED (derived)'], errors='coerce')
    flight2_30sec['TRUE AIRSPEED (derived)'] = pd.to_numeric(flight2_30sec['TRUE AIRSPEED (derived)'], errors='coerce')
    flight3_30sec['TRUE AIRSPEED (derived)'] = pd.to_numeric(flight3_30sec['TRUE AIRSPEED (derived)'], errors='coerce')
    flight4_30sec['TRUE AIRSPEED (derived)'] = pd.to_numeric(flight4_30sec['TRUE AIRSPEED (derived)'], errors='coerce')
    flight6_30sec['TRUE AIRSPEED (derived)'] = pd.to_numeric(flight6_30sec['TRUE AIRSPEED (derived)'], errors='coerce')

    # Nan waarde uit True Airspeed opvullen 
    flight1['TRUE AIRSPEED (derived)'] = flight1['TRUE AIRSPEED (derived)'].fillna(method='ffill')
    flight2['TRUE AIRSPEED (derived)'] = flight2['TRUE AIRSPEED (derived)'].fillna(method='bfill').fillna(method='ffill')
    flight3['TRUE AIRSPEED (derived)'] = flight3['TRUE AIRSPEED (derived)'].fillna(method='bfill').fillna(method='ffill')
    flight4['TRUE AIRSPEED (derived)'] = flight4['TRUE AIRSPEED (derived)'].fillna(method='bfill').fillna(method='ffill')
    flight5['TRUE AIRSPEED (derived)'] = flight5['TRUE AIRSPEED (derived)'].fillna(method='bfill').fillna(method='ffill')
    flight6['TRUE AIRSPEED (derived)'] = flight6['TRUE AIRSPEED (derived)'].fillna(method='bfill').fillna(method='ffill')
    flight7['TRUE AIRSPEED (derived)'] = flight7['TRUE AIRSPEED (derived)'].fillna(method='ffill')

    flight1_30sec['TRUE AIRSPEED (derived)'] = flight1_30sec['TRUE AIRSPEED (derived)'].fillna(method = 'ffill')

    # Geen dulplicates op waarde van tijd!

    # Nan waardes droppen 
    flight2.dropna(inplace = True)
    flight3.dropna(inplace = True)
    flight4.dropna(inplace = True)
    flight5.dropna(inplace = True)
    flight6.dropna(inplace = True)
    flight7.dropna(inplace = True)

    flight2_30sec.dropna(inplace = True)
    flight3_30sec.dropna(inplace = True)
    flight4_30sec.dropna(inplace = True)
    flight6_30sec.dropna(inplace = True)

    # Continenten worden gedefinieerd 
    zuid_amerika = ['Brazil', 'Chile', 'Colombia', 'Peru', 'Argentina', 'Ecuador', 'Paraguay', 'Bolivia', 'Suriname', 
                'French Guiana', 'Uruguay', 'Venezuela', 'Guyana', 'Falkland Islands']

    noord_amerika = ['Canada', 'United States', 'Mexico', 'Dominican Republic', 'Guatemala', 'Honduras', 'Jamaica',
                    'Nicaragua', 'Panama', 'Costa Rica', 'El Salvador', 'Haiti', 'Cuba', 'Cayman Islands', 
                    'Bahamas', 'Belize', 'Turks and Caicos Islands', 'Virgin Islands', 'Puerto Rico', 
                    'Saint Kitts and Nevis', 'Saint Lucia', 'Aruba', 'Netherlands Antilles', 'Anguilla', 
                    'Trinidad and Tobago', 'British Virgin Islands', 'Saint Vincent and the Grenadines', 
                    'Greenland', 'Bermuda', 'Montserrat','Saint Pierre and Miquelon','Midway Islands',
                    'Antigua and Barbuda','Barbados','Dominica','Martinique','Guadeloupe','Grenada','Johnston Atoll']

    azie = ['Bahrain', 'Saudi Arabia', 'Jordan', 'Lebanon', 'United Arab Emirates', 'Oman', 'Qatar', 'Japan', 
            'Azerbaijan', 'Russia', 'Tajikistan', 'Uzbekistan', 'India', 'Sri Lanka', 'Cambodia', 'Hong Kong', 
            'Thailand', 'Vietnam', 'Burma', 'Singapore', 'South Korea', 'China', 'Armenia', 'Georgia', 'Maldives', 
            'Israel', 'Afghanistan', 'Iran', 'Pakistan', 'Iraq', 'Syria', 'Kazakhstan', 'Kyrgyzstan', 'Bangladesh', 
            'Laos', 'Macau', 'Nepal', 'Bhutan', 'Brunei', 'East Timor', 'Malaysia', 'Indonesia', 'North Korea', 
            'Mongolia', 'Palestine', 'Yemen', 'Philippines', 'Taiwan','Turkmenistan','Kuwait','Myanmar','Palau','Micronesia',
            'Northern Mariana','Marshall Islands','West Bank','Northern Mariana Islands','Guam']

    afrika = ['Algeria', 'Tunisia', 'South Africa', 'Mauritius', 'Cameroon', 'Angola', 'Seychelles', 'Morocco', 
            'Ethiopia', 'Egypt', 'Sudan', 'Tanzania', 'Kenya', 'Benin', 'Burkina Faso', 'Ghana', 'Cote d\'Ivoire', 
            'Nigeria', 'Niger', 'Togo', 'Botswana', 'Congo (Brazzaville)', 'Congo (Kinshasa)', 'Swaziland', 
            'Central African Republic', 'Equatorial Guinea', 'Saint Helena', 'British Indian Ocean Territory', 
            'Zambia', 'Comoros', 'Mayotte', 'Reunion', 'Madagascar', 'Gabon', 'Sao Tome and Principe', 
            'Mozambique', 'Chad', 'Zimbabwe', 'Malawi', 'Lesotho', 'Mali', 'Gambia', 'Sierra Leone', 'Guinea-Bissau', 
            'Liberia', 'Senegal', 'Mauritania', 'Guinea', 'Cape Verde', 'Burundi', 'Somalia', 'Libya', 'Rwanda', 
            'South Sudan', 'Uganda', 'Namibia', 'Djibouti', 'Eritrea', 'Western Sahara','Juan de Nova Island']

    europa = ['Iceland', 'Belgium', 'Germany', 'Estonia', 'Finland', 'United Kingdom', 'Guernsey', 'Jersey', 
            'Isle of Man', 'Netherlands', 'Ireland', 'Denmark', 'Faroe Islands', 'Luxembourg', 'Norway', 
            'Poland', 'Sweden', 'Spain', 'Albania', 'Bulgaria', 'Cyprus', 'Croatia', 'France', 'Greece', 'Hungary', 
            'Italy', 'Slovenia', 'Czech Republic', 'Malta', 'Austria', 'Portugal', 'Bosnia and Herzegovina', 
            'Romania', 'Switzerland', 'Turkey', 'Moldova', 'Macedonia', 'Gibraltar', 'Serbia', 'Montenegro', 
            'Slovakia', 'Ukraine', 'Latvia', 'Lithuania', 'Monaco', 'Russia', 'Belarus', 'Svalbard']

    oceanie = ['Australia', 'Papua New Guinea', 'Cook Islands', 'Fiji', 'Tonga', 'Kiribati', 'Wallis and Futuna', 
            'Samoa', 'American Samoa', 'French Polynesia', 'Vanuatu', 'New Caledonia', 'New Zealand', 'Solomon Islands', 
            'Nauru', 'Tuvalu', 'Christmas Island', 'Norfolk Island', 'Cocos (Keeling) Islands', 
            'Ashmore and Cartier Islands','Niue','Wake Island']

    antarctica = ['South Georgia and the Islands','Antarctica']

    # Functie voor het bepalen van het continent
    def bepaal_continent(land):
        if land in zuid_amerika:
            return 'Zuid-Amerika'
        elif land in noord_amerika:
            return 'Noord-Amerika'
        elif land in azie:
            return 'Azië'
        elif land in afrika:
            return 'Afrika'
        elif land in europa:
            return 'Europa'
        elif land in oceanie:
            return 'Oceanië'
        elif land in antarctica:
            return 'Antarctica'

    # Landen worden aan een nieuwe kolom continenten toegevoegd
    df_airports['Continent'] = df_airports['Country'].apply(bepaal_continent)
    df_airports_schedule['Continent'] = df_airports_schedule['Country'].apply(bepaal_continent)


    # DataFrame df_airports_schedule dedupliceren op Latitude en Longitude
    df_unique = df_airports_schedule.drop_duplicates(subset=['Latitude', 'Longitude','City','Name','Continent','Country'])

    # DataFrame  df_airports dedupliceren op Latitude en Longitude
    df_airports['Name'] = df_airports['Name'].str.strip()
    df_filtered = df_airports[df_airports['Name'].str.contains('Airport', case=False, na=False)]

    # vliegvelden waarvan de Latitude en Longitude niet klopt verwijderen 
    vliegvelden_verwijderen = ["Oryol Yuzhny Airport",
                               "Kota Kinabalu Airport",
                               "Whiting Field Naval Air Station South Airport",
                               "Orlampa Inc Airport",
                               "Nasa Shuttle Landing Facility Airport",
                               "Gustaf III Airport",
                               "Guatuso Airport",
                               "Central Bolívar Airport",
                               "Comarapa Airport",
                               "Fazenda Palmital Airport",
                               "El Almendro Airport",
                               "NAS Agana Airport",
                               "Byron Airport",
                               "Cowra Airport", 
                               'Fazenda Campo Verde Airport']

    df_vliegvelden_verwijderen = df_filtered[~df_filtered['Name'].isin(vliegvelden_verwijderen)]
    df_unique_airports = df_vliegvelden_verwijderen.drop_duplicates(subset=['Latitude', 'Longitude','City','Name','Continent','Country']) 

    continent_telling_schedule = df_unique.groupby('Continent').size().reset_index(name='Aantal Vliegvelden').sort_values(by='Aantal Vliegvelden', ascending=False)
    landen_continenten_telling_schedule = df_unique.groupby(['Country']).size().reset_index(name='Aantal Vliegvelden').sort_values(by='Aantal Vliegvelden', ascending=False)
    continent_telling = df_unique_airports.groupby('Continent').size().reset_index(name='Aantal Vliegvelden').sort_values(by='Aantal Vliegvelden',ascending=False)
    landen_continenten_telling = df_unique_airports.groupby(['Country']).size().reset_index(name='Aantal Vliegvelden').sort_values(by='Aantal Vliegvelden', ascending=False)
    

    return df, daily_delay, df_cleaned, df_airports, df_unique_airports, df_unique, df_airports_schedule, df_schedule, flight1, flight1_30sec, flight2, flight2_30sec, flight3, flight3_30sec, flight4, flight4_30sec, flight5, flight5_30sec, flight6, flight6_30sec, flight7, flight7_30sec, landen_continenten_telling, continent_telling, landen_continenten_telling_schedule, continent_telling_schedule


df, daily_delay, df_cleaned, df_airports, df_unique_airports, df_unique, df_airports_schedule, df_schedule, flight1, flight1_30sec, flight2, flight2_30sec, flight3, flight3_30sec, flight4, flight4_30sec, flight5, flight5_30sec, flight6, flight6_30sec, flight7, flight7_30sec, landen_continenten_telling, continent_telling, landen_continenten_telling_schedule, continent_telling_schedule = get_data()

# Bij het gebruik van de data goed checken op fouten


mode = st.sidebar.radio("Wat wil je zien?", options=["Nieuwe app", "Gemaakte aanpassingen"], index=1)

if mode == "Gemaakte aanpassingen":
    # Veranderingen tabbladen aanmaken
    st.write(''' Welk tabblad wil je zien?''')
    tab1, tab2, tab3, tab4 = st.tabs(["layout", "Vliegvelden", "Vlucht data", "Vertragingen"])

    with tab1: 
        st.subheader('''Layout aanpassingen''')
        st.divider()
        st.image("Verandering 3.jpeg", output_format="JPEG")
        

    with tab2:
        st.subheader('''Vliegvelden aanpassingen''')
        st.divider()
        st.image("Verandering 1.jpeg", output_format="JPEG")

    with tab3:
        st.subheader('''Vlucht data aanpassingen''')
        st.divider() 
        st.image("Verandering 2.jpeg", output_format="JPEG")

    with tab4:
        st.subheader('''Vertragingen aanpassingen''')
        st.divider()
        st.image("Verandering 4.jpeg", output_format="JPEG")

if mode == "Nieuwe app":
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introductie", "Vliegvelden", "Vlucht data", "Vertragingen", "Conclusie"])
    # Zorgt ervoor dat deze code in een sidebar komt te staan


    with tab1:

        st.image("Image.jpeg", output_format="JPEG")
        st.write('''Bron: https://www.mr-online.nl/waarom-luchtvaartmaatschappijen-ondanks-het-coronavirus-vluchten-blijven-uitvoeren/''')
        multi=''' Er zijn verschillende datasets gebruikt die allemaal over vluchten gaan, deze datasets bieden gedetailleerde overzichten van vluchtgegevens, met informatie die varieert van startpunten en bestemmingen tot vluchtduur en luchtvaartmaatschappijen en hoogteverschillen.
        Met deze datasets kunnen er analyses getrokken worden over, Vertragingen, bekendste bestemmingen, Verschillen tussen snelheden van dezelfde begin tot eind bestemmingen, Afstanden berekenen tussen twee punten.
        '''
        st.markdown(multi)

    with tab2:
        # Hiermee komen de plotjes naast elkaar te staan
        left, right = st.columns([3,1]) 

        with left:
            st.write("### Vliegvelden kaart")

            # Mapbox aanmaken
            map = px.scatter_mapbox(df_unique_airports, 
                                    lat="Latitude", 
                                    lon="Longitude", 
                                    hover_name="Name", 
                                    hover_data=["Country", "Continent", "City"],
                                    color="Continent", 
                                    zoom=0.8,  
                                    height=500,
                                    width=100)
            map.update_layout(mapbox_style="open-street-map")
            map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(map, use_container_width=True)

        with right:
            # Toon het aantal vliegvelden per continent
            st.write("### Aantal vliegvelden per continent")
            st.dataframe(continent_telling[['Continent', 'Aantal Vliegvelden']]) 

        st.divider()

        # Hiermee komen de plotjes naast elkaar te staan
        left, right = st.columns([2.5,1.5]) 
        with left:  
            # Choropleth kaart aanmaken
            st.write("### Choropleth kaart met het aantal vliegvelden per land")
            fig = px.choropleth(landen_continenten_telling, locations="Country",
                                locationmode='country names',
                                color="Aantal Vliegvelden",     
                                color_continuous_scale=px.colors.sequential.Rainbow,
                                height=500,  
                                width=400) 
            st.plotly_chart(fig, use_container_width=True)

        with right:   
            # Toon de top 10 landen met de meeste vliegvelden
            st.write("### Top 10 landen met de meeste vliegvelden")  
            st.bar_chart(landen_continenten_telling[['Aantal Vliegvelden', 'Country']].head(10), y="Aantal Vliegvelden", x="Country")     

    with tab3:  
        # Functie voor het inladen van vluchtdata
        def load_clean_flight_data(selection):
            if selection == 'Vlucht 1':
                return flight1_30sec
            elif selection == 'Vlucht 2':
                return flight2_30sec
            elif selection == 'Vlucht 3':
                return flight3_30sec
            elif selection == 'Vlucht 4':
                return flight4_30sec
            elif selection == 'Vlucht 5':
                return flight5_30sec
            elif selection == 'Vlucht 6':
                return flight6_30sec
            elif selection == 'Vlucht 7':
                return flight7_30sec

        # Functie om de dichtstbijzijnde luchthaven te bepalen
        def get_nearest_airport(Lat, Lon):
            ar_afstand = []

            for i, row in df_airports.iterrows():
                Name = row['Name']
                Latitude = row['Latitude']
                Longitude = row['Longitude']

                point1 = (pd.to_numeric(Lat), pd.to_numeric(Lon))
                point2 = (pd.to_numeric(Latitude), pd.to_numeric(Longitude))
                afstand = v_direct(point1, point2)
                
                ar_afstand.append([Name, afstand])
            
            df_afstand = pd.DataFrame(ar_afstand, columns=['name', 'afstand'])
            index_afstand = df_afstand['afstand'].idxmin()
            airport_name = df_afstand.iloc[index_afstand]['name']

            return airport_name

        # Multi-select aanmaken
        selection = st.multiselect('Selecteer een vlucht', 
                                    ['Vlucht 1', 'Vlucht 2', 'Vlucht 3', 'Vlucht 4', 'Vlucht 5', 'Vlucht 6', 'Vlucht 7'],
                                    default=['Vlucht 1'])

        # Bereken het centrum voor alle geselecteerde vluchten
        all_latitudes = []
        all_longitudes = []

        for selected in selection:
            flight_data = load_clean_flight_data(selected)
            all_latitudes.extend(flight_data['Latitude'])
            all_longitudes.extend(flight_data['Longitude'])

        center_lat = mean(all_latitudes)
        center_lon = mean(all_longitudes)

        # Lijst met kleuren voor elke vlucht
        colors = ['red', 'green', 'orange', 'blue', 'purple', 'yellow', 'black']
        flight_colors = {flight: colors[i % len(colors)] for i, flight in enumerate(selection)}

        if selection and len(selection) > 0:
            m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

            for selected in selection:
                flight_data = load_clean_flight_data(selected)
                flight_color = flight_colors[selected]  # Koppel de unieke kleur

                # Markers toevoegen voor begin en eind
                start_point = flight_data.iloc[0][['Latitude', 'Longitude']].values.tolist()
                end_point = flight_data.iloc[-1][['Latitude', 'Longitude']].values.tolist()

                Latitude_begin = flight_data.iloc[0]['Latitude']
                Longitude_begin = flight_data.iloc[0]['Longitude'] 

                Latitude_eind = flight_data.iloc[-1]['Latitude']
                Longitude_eind = flight_data.iloc[-1]['Longitude']

                airport_begin = get_nearest_airport(Latitude_begin, Longitude_begin)
                airport_eind = get_nearest_airport(Latitude_eind, Longitude_eind)

                # Vlucht lijn toevoegen met de unieke kleur
                for idx, row in flight_data.iloc[1:-1].iterrows():
                    folium.CircleMarker(
                        location=[row['Latitude'], row['Longitude']],
                        radius=2,
                        color=flight_color,  # Gebruik de unieke kleur voor de vlucht
                        fill=True,
                        fill_color=flight_color,
                    ).add_to(m)

                # Start punt toevoegen
                folium.Marker(
                    start_point,
                    popup=airport_begin,
                    tooltip=flight_data.loc[flight_data.index[0]],
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(m)

                # Eind punt toevoegen
                folium.Marker(
                    end_point,
                    popup=airport_eind,
                    tooltip=flight_data.loc[flight_data.index[-1]],
                    icon=folium.Icon(color='red', icon='stop')
                ).add_to(m)
            
            st_folium(m, width=700, height=500)
        
        st.divider()

        check_box = st.checkbox("Wil je ook de hoogte verschillen per vlucht bekijken?")

        if check_box:
            # Functie maken van flight 30 sec
            def load_clean_flight_data(select):
                if select == 'Vlucht 1':
                    return flight1_30sec
                elif select == 'Vlucht 2':
                    return flight2_30sec
                elif select == 'Vlucht 3':
                    return flight3_30sec
                elif select == 'Vlucht 4':
                    return flight4_30sec
                elif select == 'Vlucht 5':
                    return flight5_30sec
                elif select == 'Vlucht 6':
                    return flight6_30sec
                elif select == 'Vlucht 7':
                    return flight7_30sec

            # Select box
            select = st.selectbox('Selecteer een vlucht', 
                                    ['Vlucht 1', 'Vlucht 2', 'Vlucht 3', 'Vlucht 4', 'Vlucht 5', 'Vlucht 6', 'Vlucht 7'])

            select_flight_data = load_clean_flight_data(select)
            def get_color_by_altitude(altitude):
                if altitude < 1000:
                    return 'blue'  # Lage altitude
                elif 1500 <= altitude < 3000:
                    return 'green'  # Medium lage altitude
                elif 3000 <= altitude < 6500:
                    return 'orange'  # Medium hoge altitude
                elif 5000 <= altitude < 10000:
                    return 'red'  # Hoge altitude
                else:
                    return 'purple' # Hoogste altitude

        # Tekst met tijden van de vluchten toevoegen
            index_tijd = select_flight_data['Time'].idxmax()
            vlucht_tijd = round(select_flight_data.iloc[index_tijd]['Time'])
            min, s = divmod(vlucht_tijd, 60)
            h, min = divmod(min, 60)

            st.write(f"De gekozen vlucht had een vluchttijd van {h} uur en {min} minuten.")

            # Folium map maken
            center_lat = select_flight_data['Latitude'].mean()
            center_lon = select_flight_data['Longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=4.3)

            # Markers toevoegen voor begin en eind
            start_point = select_flight_data.iloc[0][['Latitude', 'Longitude']].values.tolist()
            end_point = select_flight_data.iloc[-1][['Latitude', 'Longitude']].values.tolist()

            Latitude_begin = select_flight_data.iloc[0]['Latitude']
            Longitude_begin = select_flight_data.iloc[0]['Longitude'] 

            Latitude_eind = select_flight_data.iloc[-1]['Latitude']
            Longitude_eind = select_flight_data.iloc[-1]['Longitude']

                # Vlucht lijn toevoegen
            for idx, row in select_flight_data.iloc[1:-1].iterrows():
                    color = get_color_by_altitude(row['AltitudeM'])
                    folium.CircleMarker(location=[row['Latitude'], 
                                          row['Longitude']], 
                                          radius=2, 
                                          color=color, 
                                          fill=True, 
                                          fill_color= color
                                          ).add_to(m)

            folium.Marker(start_point, 
                          popup=airport_begin, 
                          tooltip=select_flight_data.loc[select_flight_data.index[0]], 
                          icon=folium.Icon(color='green', icon='play')
                          ).add_to(m)

            folium.Marker(end_point,
                          popup=airport_eind, 
                          tooltip=select_flight_data.loc[select_flight_data.index[-1]], 
                          icon=folium.Icon(color='red', icon='stop')
                          ).add_to(m)
            
            legend_html = """
            <div style="
                position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 150px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; padding:10px;">
                <h4>Altitude Kleuren:</h4>
                <i style="background: blue; width:10px; height:10px; display: inline-block;"></i> < 1000 m<br>
                <i style="background: green; width:10px; height:10px; display: inline-block;"></i> 1000 - 3000 m<br>
                <i style="background: orange; width:10px; height:10px; display: inline-block;"></i> 3000 - 5000 m<br>
                <i style="background: red; width:10px; height:10px; display: inline-block;"></i> 5000 - 10000 m<br>
                <i style="background: purple; width:10px; height:10px; display: inline-block;"></i> > 10000 m
            </div>
            """
            
            # Voeg de legenda toe aan de kaart
            m.get_root().html.add_child(folium.Element(legend_html))

            # Laad de kaart met Streamlit
            st.components.v1.html(m._repr_html_(), width=700, height=700)

            #  Tekst toe voegen met hoogte per vlucht
            index_altitude = select_flight_data['AltitudeM'].idxmax()
            vlucht_altitude = round(select_flight_data.iloc[index_altitude]['AltitudeM'])
            
            st.write(f"De maximale hoogte in meters die deze vlucht heeft gehaald is, {vlucht_altitude} meters.")
            st.line_chart(select_flight_data['AltitudeM'])

        st.write(f'''Hieruit is te zien dat de zeven vluchten allemaal verschillende vliegpaden hebben, want geen één vlucht kan precies hetzelfde vliegpad hebben. Daarnaast is het ook te zien dat de vluchten verschillende hoogtes hadden tijdens het opstijgen, het vliegniveau en het landen. 
                Daarnaast wordt het duidelijk geen vlucht dezelfde vliegtijd heeft, dat kan komen door verschillende weersomstandigheden, zoals wind mee en wind tegen.''')


    with tab4:
        st.header('Voorspelling van vluchtvertragingen')
        X_data = np.array(pd.to_datetime(daily_delay.index).astype('int64').values.reshape(-1, 1)) 
        vertraging_waarden = daily_delay.values

        # Regressiemodel
        voorspel_model = LinearRegression()
        voorspel_model.fit(X_data, vertraging_waarden)

        # Voorspellingen maken
        toekomst_data = pd.date_range(start='2019-01-01', end='2021-12-31')
        toekomst = np.array(toekomst_data.astype('int64').values.reshape(-1, 1))  
        vertraging_voorspelling = voorspel_model.predict(toekomst)

        
        voorspelling_df = pd.DataFrame({
            'Datum': toekomst_data,
            'Verwachte Vertraging': vertraging_voorspelling})

        actuele_data_df = pd.DataFrame({
            'Datum': daily_delay.index,
            'Gemiddelde Vertraging': daily_delay.values})

        # Plot maken van gemiddelde vertraging en voorspelling
        fig = px.line(actuele_data_df, x='Datum',  y='Gemiddelde Vertraging', title='Dagelijkse gemiddelde vertragingen tussen 2019 en 2020', markers=True, labels={'Gemiddelde Vertraging': 'Gemiddelde Vertraging (Minuten)'})

        # Voorspelling toevoegen aan de plot
        fig.add_scatter(x=voorspelling_df['Datum'], y=voorspelling_df['Verwachte Vertraging'], mode='lines', name='Voorspelling', line=dict(color='red', dash='dash'))

        # Lay-out van de plot instellen
        fig.update_layout(
            xaxis_title='Datum', 
            yaxis_title='Vertraging (Minuten)', 
            xaxis_tickangle=-45,
            width=800, 
            height=600
        )

        # Plot weergeven in Streamlit
        st.plotly_chart(fig)

        # Berekenen van de R-squared waarde als prestatiemaatstaf voor het model
        model_r_squared = voorspel_model.score(X_data, vertraging_waarden)
        st.write(f"R-squared van het model: {model_r_squared:.4f}")

        st.divider()

        left, right = st.columns([2, 2]) 
        
        with left:
            st.write("### Vertraging van vliegtuigen op Schiphol in 2019")
            st.scatter_chart(df_airports_schedule.loc[(df_airports_schedule["STD"] <= "31/12/2019") & (df_airports_schedule["Name"] == "Amsterdam Airport Schiphol") & (df_airports_schedule["Delay"] <= 15000)], x="STD", y="Delay"#, x_label="Datum", y_label="Vertraging in seconden"
                            )

        with right:    
            st.write("### Vertraging van vliegtuigen op Schiphol in 2020")
            st.scatter_chart(df_airports_schedule.loc[(df_airports_schedule["STD"] > "31/12/2019") & (df_airports_schedule["Name"] == "Amsterdam Airport Schiphol") & (df_airports_schedule["Delay"] <= 15000)], x="STD", y="Delay"#, x_label="Datum", y_label="Vertraging in seconden"
                            )
        
        st.divider()

        @st.cache_data
        def get_land_data(land):
            return merged_data[merged_data['Country'] == land]

        # Voeg een maandkolom toe aan de data voor maandelijkse aggregatie
        df_cleaned['maand'] = df_cleaned['STD'].dt.to_period('M')

        # Data set samenvoegen
        @st.cache_data
        def merge_data(data1, data2, left, right, join_type):
        
            samengevoegd = pd.merge(data1, data2, left_on=left, right_on=right, how=join_type)
            return samengevoegd

        merged_data = merge_data(df_airports, df_cleaned, 'ICAO', 'Org/Des', 'inner')

        # Keuzemenu voor land
        land_keuze = merged_data['Country'].unique()
        land = st.selectbox('Welk land wil je zien?', land_keuze)

        # Filteren op het geselecteerde land
        data_land = get_land_data(land)

        # Filteren op uitgaande vluchten
        vluchten = data_land[data_land['LSV'] == 'S']
        maand_data = vluchten.groupby('maand').agg({
            'Delay_Minutes': 'mean',
            'FLT': 'count'
        }).rename(columns={'Delay_Minutes': 'Gem_Vertraging', 'FLT': 'Vluchten'})

        maand_data['tijd_index'] = range(1, len(maand_data) + 1)

        # Regressiemodel 
        X = maand_data['Vluchten'].values.reshape(-1, 1)
        y = maand_data['Gem_Vertraging'].values
        model = LinearRegression()
        model.fit(X, y)

        # Figuur aan maken
        fig = px.line(
            maand_data,
            x=maand_data.index.astype(str),
            y='Gem_Vertraging',
            title=f'Vertraging van vluchten vanuit {land} tussen 2019 tot 2020',
            labels={'Gem_Vertraging': 'Vertraging in minuten', 'index': 'Maand'},
            markers=True
        )

        # Lay-out verbeteren
        fig.update_layout(
            xaxis_title='Maand',
            yaxis_title='Vertraging in minuten',
            xaxis_tickangle=-45,
            template='plotly_white',
            title_font=dict(size=16),
            xaxis=dict(tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=12)),
            showlegend=False
        )

        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')

        # Grafiek in Streamlit
        st.plotly_chart(fig, use_container_width=True)


        st.write(f"""
            In april 2020 kwam de luchtvaart grotendeels stil te liggen door COVID-19.
            Dit is duidelijk zichtbaar in de grafieken. Waar de grafiek van 2020 in
            de eerste maanden nog de trend van 2019 volgt met betrekking tot
            vertragingen, zien we vanaf april een sterke daling. Dit komt voornamelijk
            doordat er toen veel minder vluchten waren, en daardoor ook minder vertragingen.
            In juli 2020 begint de luchtvaart weer op gang te komen, en daarmee nemen ook de
            vertragingen direct toe, zowel in aantal als in omvang. De verwachting is
            dat, naarmate de gevolgen van COVID-19 afnemen, de grafiek weer meer zal gaan lijken
            op die van 2019.""")
        
    

        with tab5: 
            st.write(f"""
            De analyses van de vluchtgegevens heeft waardevolle inzichten opgeleverd in de prestaties en trends binnen de luchtvaartsector. Door de gegevens te onderzoeken, hebben we een aantal belangrijke bevindingen kunnen identificeren:
        -	Het identificeren van alle vliegvelden over de hele wereld, hierdoor kun je goed zien welke landen/ continenten de meeste vliegvelden hebben. 
        -	Het identificeren dat er snelheid en hoogte verschillen zitten in vluchten met dezelfde begin- en eindbestemming. 
        -	Het identificeren van het aantal vliegtuigen op een vliegveld. 
        -   Het identificeren van vertragingen per land.

        Al met al bieden deze datasets een rijke bron van informatie over de luchtvaartsector. De inzichten die zijn opgedaan kunnen bijdragen aan effectievere strategieën voor luchtvaartmaatschappijen.""")
        
     