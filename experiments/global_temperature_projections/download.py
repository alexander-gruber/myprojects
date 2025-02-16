import cdsapi

# CMIP6 climate projections
# https://cds.climate.copernicus.eu/datasets/projections-cmip6?tab=download

# dataset = "projections-cmip6"
# request = {
#     "temporal_resolution": "monthly",
#     "experiment": "historical",
#     "variable": "air_temperature",
#     "model": "access_cm2",
#     "year": [
#         "2000", "2001", "2002",
#         "2003", "2004", "2005",
#         "2006", "2007", "2008",
#         "2009", "2010", "2011",
#         "2012", "2013", "2014"
#     ],
#     "month": [
#         "01", "02", "03",
#         "04", "05", "06",
#         "07", "08", "09",
#         "10", "11", "12"
#     ]
# }
#
# client = cdsapi.Client()
# client.retrieve(dataset, request).download()



# Gridded monthly climate projection dataset underpinning the IPCC AR6 Interactive Atlas
# https://cds.climate.copernicus.eu/datasets/projections-climate-atlas?tab=download

dataset = "projections-climate-atlas"
request = {
    "origin": "cmip6",
    "experiment": "historical",
    "domain": "global",
    "period": "1850-2014",
    "variable": "monthly_mean_of_daily_mean_temperature"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()

