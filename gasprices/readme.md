dataset structure

gas_prices
Column	Type	Comment
id	bigint Auto Increment [nextval('gas_prices_id_seq')]	
gas_station_id	bigint	
fuel_type	character varying(255) NULL	
price	integer NULL	
created_at	timestamp(0) NULL	
updated_at	timestamp(0) NULL	

gas_stations
Column	Type	Comment
id	bigint Auto Increment [nextval('gas_stations_id_seq')]	
company_name	character varying(255) NULL	
site	text NULL	
country	character varying(255) NULL	
city	character varying(255) NULL	
street_name	character varying(255) NULL	
house_number	character varying(255) NULL	
postal_code	character varying(255) NULL	
lat	double precision NULL	
lng	double precision NULL	
is_favorite	boolean [false]	
created_at	timestamp(0) NULL	
updated_at	timestamp(0) NULL	
location	geometry(Point,4326) NULL	
external_id	character varying(255) NULL	


# Commands used to create environment
```
python3 -m venv .venv
```
```
source .venv/bin/activate
```
```
pip freeze > requirements.txt
```

# Commands to run after cloning repo
## Prerequisites
```
apt install python3.13-venv
```
## setup local python environment
```
python3 -m venv .venv
```
```
source .venv/bin/activate
```
```
pip install -r requirements.txt
```
## setup postgres database
```
docker compose up -d
```
```
docker cp puzzle.dump gasprices-postgres-1:. 
```
```
docker exec -i gasprices-postgres-1 createdb -U postgres puzzle
```
```
docker exec -i gasprices-postgres-1 pg_restore -U postgres -d puzzle < puzzle.dump
```

