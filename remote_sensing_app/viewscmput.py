from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sentinelhub import SentinelHubRequest, SentinelHubCatalog, DataCollection, MimeType, CRS, SHConfig, BBox
from shapely.geometry import shape, Point, mapping, Polygon, MultiPolygon
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
import geopandas as gpd
from .serializers import EndDateSerializer, IndicesSerializer
import json
import pandas as pd
from prophet import Prophet
import random
import os

config = SHConfig()
config.sh_client_id = 'cf7c454e-772c-4a94-9f24-5fb1348530fa'
config.sh_client_secret = 'Bm6yTh2qbpUHMTVgERIBOx0e6ksPOKk3'


# Create a JSON file to store locations
location_file_path = 'location_data.json'

def get_data_collection(satellite):
    """
    Returns the DataCollection object based on the satellite name.
    :param satellite: String representing the satellite name.
    :return: Corresponding DataCollection object or None if unsupported.
    """
    satellite = satellite.lower()
    if satellite == "sentinell2a":
        return DataCollection.SENTINEL2_L2A
    elif satellite == "sentinell1c":
        return DataCollection.SENTINEL2_L1C
    elif satellite == "landsatl2":
        return DataCollection.LANDSAT_OT_L2
    elif satellite == "landsatl1":
        return DataCollection.LANDSAT_OT_L1
    else:
        return None

def set_config(satellite):
    satellite = satellite.lower()
    if satellite == "landsatl2" or satellite == "landsatl1":
        return "https://services-uswest2.sentinel-hub.com"
    else:
        return "https://services.sentinel-hub.com"


class SaveLocationView(APIView):
    def post(self, request):
        if not request.data.get('latitude') or not request.data.get('longitude'):
            return JsonResponse({'error': 'Latitude and longitude are required.'}, status=status.HTTP_400_BAD_REQUEST)

        latitude = request.data['latitude']
        longitude = request.data['longitude']

        try:
            # Append new location data to JSON file
            with open(location_file_path, 'a') as file:
                json.dump({'latitude': latitude, 'longitude': longitude, 'country' : request.data['country_name'], 'region' :request.data['region'], 'city' : request.data['city']}, file)
                file.write('\n')  # Write a new line for the next entry

            return JsonResponse({'status': 'success', 'data': {'latitude': latitude, 'longitude': longitude}}, status=status.HTTP_201_CREATED)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class GetLocationView(APIView):
    def get(self, request):
        try:
            locations = []
            if os.path.exists(location_file_path):
                with open(location_file_path, 'r') as file:
                    for line in file:
                        locations.append(json.loads(line.strip()))
                return JsonResponse({'status': 'success', 'locations': locations}, status=status.HTTP_200_OK)
            else:
                return JsonResponse({'error': 'Location data not found.'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



class SentinelDataAvailabilityView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = EndDateSerializer(data=request.data)
        if serializer.is_valid():
            end_date = serializer.validated_data['end_date']
            cloud_coverage = int(request.data.get('cloud_coverage', 100))

            if cloud_coverage > 100:
                return Response({'error': 'Cloud coverage value cannot be greater than 100.'}, status=status.HTTP_400_BAD_REQUEST)

            # Determine the satellite
            satellite = request.data.get('satellite', 'sentinell2a')
            config.sh_base_url  = set_config(satellite)
            data_collection = get_data_collection(satellite)
            if not data_collection:
                return Response({'error': 'Unsupported satellite type.'}, status=status.HTTP_400_BAD_REQUEST)

            catalog = SentinelHubCatalog(config=config)
            search_results = catalog.search(
                collection=data_collection,
                bbox=bbox,
                filter=f"eo:cloud_cover < {cloud_coverage}",
                time=('2016-01-01', end_date),
            )
            
            # Prepare a list of valid dates with available data
            available_dates = set()

            for result in search_results:
                try:
                    # Check for data availability for the given date
                    date = datetime.fromisoformat(result['properties']['datetime'][:-1]).date().isoformat()
                    available_dates.add(date)
                except KeyError:
                    # Handle cases where the datetime or other properties may not exist
                    continue
                except ValueError:
                    # Handle any date parsing errors
                    continue

            if not available_dates:
                return Response({'error': 'No data available for the specified time range.'}, status=status.HTTP_404_NOT_FOUND)

            return JsonResponse(list(available_dates), safe=False)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Soil quality
class SoilQualityView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return { 
                    input: ["B08", "B11", "B12", "SCL"], 
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }
            
            function validate(sample) {
                var scl = sample.SCL;
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false;
                }
                return true;
            }
            
            function evaluatePixel(samples, scenes) {
                var validValuesB08 = [], validValuesB11 = [], validValuesB12 = [];
                
                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B11 > 0 && sample.B12 > 0) {
                        if (validate(sample)) {
                            validValuesB08.push(sample.B08);
                            validValuesB11.push(sample.B11);
                            validValuesB12.push(sample.B12);
                        }
                    }
                }
                
                if (validValuesB08.length > 0) {
                    var avgB08 = validValuesB08.reduce((a, b) => a + b, 0) / validValuesB08.length;
                    var avgB11 = validValuesB11.reduce((a, b) => a + b, 0) / validValuesB11.length;
                    var avgB12 = validValuesB12.reduce((a, b) => a + b, 0) / validValuesB12.length;
                    
                    var sqi = (avgB11 - avgB08) / (avgB11 + avgB08 + avgB12);
                } else {
                    sqi = -9999;
                }
                
                return [sqi];
            }
            """
            
            satellite = request.data.get('satellite', 'sentinell2a')
            config.sh_base_url = set_config(satellite)
            data_collection = get_data_collection(satellite)

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[SentinelHubRequest.input_data(data_collection=data_collection, time_interval=(date, date))],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            if np.all(response == -9999):
                return Response({'error': 'No valid data available for the given date and area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])
            
            def reclassify_sqi(sqi_array):
                classified_array = np.zeros_like(sqi_array, dtype=np.uint8)
                classified_array[(sqi_array <= -0.2) & (sqi_array != -9999)] = 1  # Very poor soil quality
                classified_array[(sqi_array > -0.2) & (sqi_array <= 0)] = 2  # Poor soil quality
                classified_array[(sqi_array > 0) & (sqi_array <= 0.2)] = 3  # Moderate soil quality
                classified_array[(sqi_array > 0.2) & (sqi_array <= 0.4)] = 4  # Good soil quality
                classified_array[(sqi_array > 0.4)] = 5  # Excellent soil quality
                classified_array[(sqi_array == -9999)] = 0  # No data
                return classified_array

            classified_image = reclassify_sqi(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)
            
            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries]
            geojson_data = {"type": "FeatureCollection", "features": features}
            
            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()
            
            return JsonResponse(json.loads(intersection_geojson))
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Vegetation Health
class NDVIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return { 
                    input: ["B04", "B08", "B11", "SCL"],  // Added B11 for better vegetation detection under trees
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Exclude cloud and other invalid pixels, keep tree canopy (SCL = 4) for processing
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude clouds, cloud shadows, and water
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB08 = [], validValuesB04 = [], validValuesB11 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B04 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB08[a] = sample.B08;
                            validValuesB04[a] = sample.B04;
                            validValuesB11[a] = sample.B11;  // Using SWIR for under-canopy plants
                            a++;
                        }
                    }
                }

                var ndvi;
                if (a > 0) {
                    var avgB08 = getValue(validValuesB08);
                    var avgB04 = getValue(validValuesB04);
                    var avgB11 = getValue(validValuesB11);
                    
                    // Modified NDVI calculation to include under-canopy plants using B11
                    if (avgB11 > 0.3) { // Threshold to identify tree canopies
                        // Adjust NDVI for plants under tree canopies
                        ndvi = (avgB08 - avgB04) / (avgB08 + avgB04 + avgB11); 
                    } else {
                        // Regular NDVI calculation
                        ndvi = (avgB08 - avgB04) / (avgB08 + avgB04);
                    }
                } else {
                    ndvi = -9999; // No valid data
                }

                return [ndvi];
            }
            """
            satellite = request.data.get('satellite', 'sentinell2a')
            config.sh_base_url  = set_config(satellite)
            data_collection = get_data_collection(satellite)

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[SentinelHubRequest.input_data(data_collection=data_collection, time_interval=(date, date))],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            # Check if the response is empty (all invalid values)
            if np.all(response == -9999):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_ndvi(ndvi_array):
                classified_array = np.zeros_like(ndvi_array, dtype=np.uint8)
                classified_array[(ndvi_array <= 0) & (ndvi_array != -9999)] = 1
                classified_array[(ndvi_array > 0) & (ndvi_array <= 0.1)] = 2
                classified_array[(ndvi_array > 0.1) & (ndvi_array <= 0.2)] = 3
                classified_array[(ndvi_array > 0.2) & (ndvi_array <= 0.4)] = 4
                classified_array[(ndvi_array > 0.4) & (ndvi_array <= 0.5)] = 5
                classified_array[(ndvi_array > 0.5) & (ndvi_array <= 0.6)] = 6
                classified_array[(ndvi_array > 0.6) & (ndvi_array <= 0.7)] = 7
                classified_array[(ndvi_array > 0.7) & (ndvi_array <= 1)] = 8
                classified_array[(ndvi_array == -9999)] = 0  # Any missing or cloudy data
                return classified_array

            classified_image = reclassify_ndvi(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


#COFFEE RIPENESS USING NIR Updated with canopy detection for all
class NIRView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        date = request.data.get('date')
        if not date:
            return Response({'error': 'Date is required.'}, status=status.HTTP_400_BAD_REQUEST)

        evalscript = """
        function setup() {
            return { 
                input: ["B08", "B04", "B03", "SCL"],  // B08 for NIR, B04 for Red, B03 for Green
                output: { bands: 2, sampleType: "FLOAT32" }, 
                mosaicking: "ORBIT" 
            };
        }

        function validate(sample) {
            var scl = sample.SCL;
            // Exclude cloud and other invalid pixels
            if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                return false; // Exclude clouds, shadows, water, and other invalid pixels
            }
            return true;
        }

        function evaluatePixel(samples) {
            var validValuesB08 = [], validValuesB04 = [], validValuesB03 = [];
            var a = 0;

            for (var i = 0; i < samples.length; i++) {
                var sample = samples[i];
                if (sample.B08 > 0 && sample.B04 > 0 && sample.B03 > 0) {
                    var isValid = validate(sample);
                    if (isValid) {
                        validValuesB08[a] = sample.B08;  // NIR
                        validValuesB04[a] = sample.B04;  // Red
                        validValuesB03[a] = sample.B03;  // Green
                        a++;
                    }
                }
            }

            var nirValue = -9999; // Default if no valid data
            var canopyFlag = 0;  // Default flag for canopy detection
            if (a > 0) {
                // Compute the average NIR value
                nirValue = validValuesB08.reduce((a, b) => a + b, 0) / a;  // Average NIR
                
                // Detect canopy presence based on relative band ratios (example logic)
                var ndvi = (nirValue - validValuesB04.reduce((a, b) => a + b, 0) / a) / 
                           (nirValue + validValuesB04.reduce((a, b) => a + b, 0) / a);
                
                if (ndvi > 0.4 && validValuesB03.reduce((a, b) => a + b, 0) / a < 0.3) {
                    canopyFlag = 1;  // Flag for canopy-covered vegetation
                }
            }

            return [nirValue, canopyFlag];
        }
        """

        sentinel_request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
            responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
            bbox=bbox,
            size=[512, 354.253],
            config=config,
        )

        response = sentinel_request.get_data()[0]

        # Check if the response is empty (all invalid values)
        if np.all(response == -9999):
            return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

        transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

        def reclassify_nir_canopy(data_array):
            nir_array, canopy_array = data_array[..., 0], data_array[..., 1]
            classified_array = np.zeros_like(nir_array, dtype=np.uint8)
            
            # Reclassify NIR values for ripeness detection
            classified_array[(nir_array <= 0.1) & (nir_array != -9999)] = 1  # Unripe
            classified_array[(nir_array > 0.1) & (nir_array <= 0.3)] = 2  # Almost ripe
            classified_array[(nir_array > 0.3) & (nir_array <= 0.5)] = 3  # Ripe
            classified_array[(nir_array > 0.5) & (nir_array <= 0.7)] = 4  # Overripe
            classified_array[(nir_array > 0.7) & (nir_array <= 1)] = 5    # Very overripe

            # Include canopy detection as a separate class
            classified_array[(canopy_array == 1)] = 6  # Canopy-covered vegetation
            
            classified_array[(nir_array == -9999)] = 0  # Set invalid pixels to 0
            return classified_array

        classified_image = reclassify_nir_canopy(response)
        shapes_gen = shapes(classified_image, mask=None, transform=transform)
        geometries = list(shapes_gen)

        features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
        geojson_data = {"type": "FeatureCollection", "features": features}

        geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
        geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
        intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
        intersection_geojson = intersection_df.to_json()

        return JsonResponse(json.loads(intersection_geojson))


# Humidity level
class NDWIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return { 
                    input: ["B03", "B08", "B11", "SCL"],  // Added B11 for detecting vegetation under trees
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Using SCL to filter out clouds, shadows, and invalid pixels
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude cloud and cloud shadow pixels
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB03 = [], validValuesB08 = [], validValuesB11 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B03 > 0 && sample.B08 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB03[a] = sample.B03;
                            validValuesB08[a] = sample.B08;
                            validValuesB11[a] = sample.B11;  // B11 added for under-tree vegetation detection
                            a++;
                        }
                    }
                }

                var ndwi;
                if (a > 0) {
                    var avgB03 = getValue(validValuesB03);
                    var avgB08 = getValue(validValuesB08);
                    var avgB11 = getValue(validValuesB11);

                    if (avgB11 > 0.3) {  // Threshold to account for vegetation under tree canopies
                        // Adjust NDWI considering B11 for under-canopy water
                        ndwi = (avgB03 - avgB08) / (avgB03 + avgB08 + avgB11);
                    } else {
                        // Standard NDWI calculation
                        ndwi = (avgB03 - avgB08) / (avgB03 + avgB08);
                    }
                } else {
                    ndwi = -9999; // No valid data
                }

                return [ndwi];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                ],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            # Check if the response is empty (all invalid values)
            if np.all(response == 0):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_ndwi(ndwi_array):
                classified_array = np.zeros_like(ndwi_array, dtype=np.uint8)
                classified_array[(ndwi_array <= -1) & (ndwi_array != -9999)] = 1
                classified_array[(ndwi_array > -1) & (ndwi_array <= 0)] = 2
                classified_array[(ndwi_array > 0) & (ndwi_array <= 0.1)] = 3
                classified_array[(ndwi_array > 0.1) & (ndwi_array <= 0.2)] = 4
                classified_array[(ndwi_array > 0.2) & (ndwi_array <= 0.3)] = 5
                classified_array[(ndwi_array > 0.3) & (ndwi_array <= 0.4)] = 6
                classified_array[(ndwi_array > 0.4) & (ndwi_array <= 0.5)] = 7
                classified_array[(ndwi_array > 0.5) & (ndwi_array <= 1)] = 8
                classified_array[(ndwi_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_ndwi(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Plant Moisture
class NDMIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return { 
                    input: ["B08", "B11", "B04", "SCL"],  // Added B04 for vegetation detection under trees
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Using SCL to filter out clouds and invalid pixels, keeping tree canopy (SCL = 4) for analysis
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude clouds, cloud shadows, and water
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB08 = [], validValuesB11 = [], validValuesB04 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B11 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB08[a] = sample.B08;
                            validValuesB11[a] = sample.B11;
                            validValuesB04[a] = sample.B04;  // Added B04 for better vegetation assessment
                            a++;
                        }
                    }
                }

                var ndmi;
                if (a > 0) {
                    var avgB08 = getValue(validValuesB08);
                    var avgB11 = getValue(validValuesB11);
                    var avgB04 = getValue(validValuesB04);

                    if (avgB04 > 0.3) { // Threshold to detect vegetation under tree canopies
                        // Adjust NDMI for plants under tree canopies
                        ndmi = (avgB08 - avgB11) / (avgB08 + avgB11 + avgB04); 
                    } else {
                        // Regular NDMI calculation
                        ndmi = (avgB08 - avgB11) / (avgB08 + avgB11);
                    }
                } else {
                    ndmi = -9999; // No valid data
                }

                return [ndmi];
            }
            """
            
            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                ],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            # Check if the response is empty (all invalid values)
            if np.all(response == 0):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_ndmi(ndmi_array):
                classified_array = np.zeros_like(ndmi_array, dtype=np.uint8)
                classified_array[(ndmi_array <= -1) & (ndmi_array != -9999)] = 1
                classified_array[(ndmi_array > -1) & (ndmi_array <= 0)] = 2
                classified_array[(ndmi_array > 0) & (ndmi_array <= 0.1)] = 3
                classified_array[(ndmi_array > 0.1) & (ndmi_array <= 0.2)] = 4
                classified_array[(ndmi_array > 0.2) & (ndmi_array <= 0.3)] = 5
                classified_array[(ndmi_array > 0.3) & (ndmi_array <= 0.4)] = 6
                classified_array[(ndmi_array > 0.4) & (ndmi_array <= 0.5)] = 7
                classified_array[(ndmi_array > 0.5) & (ndmi_array <= 1)] = 8
                classified_array[(ndmi_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_ndmi(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Coffee Ripeness
class CRIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)

        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return { 
                    input: ["B04", "B11", "SCL"],  // Included B11 for detecting vegetation under trees
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Exclude cloud and other invalid pixels, keep tree canopy (SCL = 4) for processing
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude clouds, cloud shadows, and water
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB04 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B04 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB04[a] = sample.B04;
                            a++;
                        }
                    }
                }

                var cri;
                if (a > 0) {
                    cri = getValue(validValuesB04);
                } else {
                    cri = -9999; // No valid data
                }

                return [cri];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                ],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            # Check if the response is empty (all invalid values)
            if np.all(response == -9999):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_cri(cri_array):
                classified_array = np.zeros_like(cri_array, dtype=np.uint8)
                classified_array[(cri_array <= 10) & (cri_array != -9999)] = 1
                classified_array[(cri_array > 10) & (cri_array <= 20)] = 2
                classified_array[(cri_array > 20) & (cri_array <= 30)] = 3
                classified_array[(cri_array > 30) & (cri_array <= 40)] = 4
                classified_array[(cri_array > 40) & (cri_array <= 50)] = 5
                classified_array[(cri_array > 50) & (cri_array <= 60)] = 6
                classified_array[(cri_array > 60) & (cri_array <= 70)] = 7
                classified_array[(cri_array > 70) & (cri_array <= 100)] = 8
                classified_array[(cri_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_cri(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Ground Temperature
class LSTView(APIView):
    def post(self, request, format=None):
        # Extract GeoJSON polygon from the request
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        # Define bounding box based on the polygon
        bbox = polygon.bounds  # (minx, miny, maxx, maxy)
        bbox = BBox(bbox=(bbox[0], bbox[1], bbox[2], bbox[3]), crs=CRS.WGS84)
      
        # Create serializer instance with request data
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            # Extract validated data
            date = serializer.validated_data['date']
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        evalscript = """
            var option = 0;
            var minC = 0;
            var maxC = 50;
            var NDVIs = 0.2;
            var NDVIv = 0.8;
            var waterE = 0.991;
            var soilE = 0.966;
            var vegetationE = 0.973;
            var C = 0.009;
            var bCent = 0.000010854;
            var rho = 0.01438;

            if (option == 2) {
                minC = 0;
                maxC = 25;
            }

            let viz = ColorGradientVisualizer.createRedTemperature(minC, maxC);

            function setup() {
                return {
                    input: [
                        { datasource: "S3SLSTR", bands: ["S8"] },
                        { datasource: "S3OLCI", bands: ["B06", "B08", "B11"] } // Include B11 for vegetation detection under trees
                    ],
                    output: [
                        { id: "default", bands: 3, sampleType: SampleType.AUTO }
                    ],
                    mosaicking: "ORBIT"
                };
            }

            // Cloud validation function
            function validate(sample) {
                var scl = sample.SCL;
                // Using SCL to filter out clouds and invalid pixels
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude cloud and cloud shadow pixels
                }
                return true;
            }

            // Function to calculate Land Surface Emissivity (LSE)
            function LSEcalc(NDVI, Pv) {
                var LSE;
                if (NDVI < 0) {
                    LSE = waterE;
                } else if (NDVI < NDVIs) {
                    LSE = soilE;
                } else if (NDVI > NDVIv) {
                    LSE = vegetationE;
                } else {
                    LSE = vegetationE * Pv + soilE * (1 - Pv) + C;
                }
                return LSE;
            }

            function evaluatePixel(samples) {
                var validLSTs = [];
                var invalidLSTs = [];
                var N = samples.S3SLSTR.length;

                for (let i = 0; i < N; i++) {
                    var sampleSLSTR = samples.S3SLSTR[i];
                    var sampleOLCI = samples.S3OLCI[i];

                    var Bi = sampleSLSTR.S8;
                    var B06i = sampleOLCI.B06;
                    var B08i = sampleOLCI.B08;
                    var B11i = sampleOLCI.B11; // Added B11 for better vegetation assessment

                    if ((Bi <= 173 || Bi >= 65000) || (B06i <= 0 || B08i <= 0 || B11i <= 0)) {
                        continue; // Skip invalid measurements
                    }

                    var isValid = validate(sampleOLCI); // Validate using the cloud mask (SCL)
                    var S8BTi = Bi - 273.15; // Convert to Celsius
                    var NDVIi = (B08i - B11i) / (B08i + B11i);

                    // Detect vegetation under tree canopies
                    if (B11i > 0.3) {
                        NDVIi = (B08i - (B11i / 2)) / (B08i + (B11i / 2)); // Adjust NDVI for canopy vegetation
                    }

                    var PVi = Math.pow(((NDVIi - NDVIs) / (NDVIv - NDVIs)), 2);
                    var LSEi = LSEcalc(NDVIi, PVi);
                    var LSTi = (S8BTi / (1 + (((bCent * S8BTi) / rho) * Math.log(LSEi))));

                    if (isValid) {
                        validLSTs.push(LSTi);
                    } else {
                        invalidLSTs.push(LSTi);
                    }
                }

                // Select valid LSTs if available, otherwise fall back to invalid ones
                var LSTsToUse = validLSTs.length > 0 ? validLSTs : invalidLSTs;

                var outLST;
                if (option == 0) {
                    outLST = LSTsToUse.reduce((a, b) => a + b, 0) / LSTsToUse.length; // Average
                } else if (option == 1) {
                    outLST = Math.max(...LSTsToUse); // Maximum
                } else {
                    var avg = LSTsToUse.reduce((a, b) => a + b, 0) / LSTsToUse.length;
                    outLST = Math.sqrt(LSTsToUse.reduce((sum, lst) => sum + Math.pow(lst - avg, 2), 0) / (LSTsToUse.length - 1)); // Standard deviation
                }

                return viz.process(outLST);
            }
        """

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL3_SLSTR,
                    identifier="S3SLSTR",
                    time_interval=(date, date),
                ),
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL3_OLCI,
                    identifier="S3OLCI",
                    time_interval=(date, date),
                ),
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF),
            ],
            bbox=bbox,
            size=[512, 354.253],
            config=config,
        )

        response = request.get_data()
        response_data = response[0]

        transform = rasterio.transform.from_bounds(*bbox, response_data.shape[1], response_data.shape[0])

        # Scaling factors
        minC = 0  # Set based on your evalscript
        maxC = 50  # Set based on your evalscript

        # Convert red channel values to temperature
        response_data_temp = minC + (response_data[..., 0] / 255.0) * (maxC - minC)

        # Check if any value in red channel is 255
        if np.any(response_data[..., 0] == 255):
            return Response({'error': 'Red channel value is 255, possibly querying temperature for tomorrow.'}, status=status.HTTP_400_BAD_REQUEST)

        # Calculate mean of response_data_temp and round to nearest integer
        mean_temp = np.round(np.mean(response_data_temp))

        return Response({'mean_temperature': int(mean_temp)}, status=status.HTTP_200_OK)


#Water Stress
class WaterStressIndexView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return {
                    input: ["B04", "B08", "B11", "SCL"], // Added B11 for under-canopy vegetation
                    output: { bands: 1, sampleType: "FLOAT32" },
                    mosaicking: "ORBIT"
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000); // Filter by recent data
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Exclude cloud and invalid pixels; keep tree canopy (SCL = 4) for processing
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude clouds, cloud shadows, and water
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB08 = [], validValuesB04 = [], validValuesB11 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B04 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB08[a] = sample.B08;
                            validValuesB04[a] = sample.B04;
                            validValuesB11[a] = sample.B11;  // Include SWIR for under-canopy detection
                            a++;
                        }
                    }
                }

                var wst;
                if (a > 0) {
                    // Calculate WST using valid pixel data
                    var avgB04 = getValue(validValuesB04);
                    var avgB08 = getValue(validValuesB08);
                    var avgB11 = getValue(validValuesB11);

                    // Modified WST calculation to include under-canopy plants using B11
                    wst = (avgB04 - avgB08) / (avgB04 + avgB08 + avgB11);
                } else {
                    wst = -9999; // No valid data
                }

                return [wst];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                ],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            # Check if the response is empty (all invalid values)
            if np.all(response == -9999):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_wsi(wsi_array):
                classified_array = np.zeros_like(wsi_array, dtype=np.uint8)
                classified_array[(wsi_array <= -1) & (wsi_array != -9999)] = 1
                classified_array[(wsi_array > -1) & (wsi_array <= 0)] = 2
                classified_array[(wsi_array > 0) & (wsi_array <= 0.1)] = 3
                classified_array[(wsi_array > 0.1) & (wsi_array <= 0.2)] = 4
                classified_array[(wsi_array > 0.2) & (wsi_array <= 0.3)] = 5
                classified_array[(wsi_array > 0.3) & (wsi_array <= 0.4)] = 6
                classified_array[(wsi_array > 0.4) & (wsi_array <= 0.5)] = 7
                classified_array[(wsi_array > 0.5) & (wsi_array <= 1)] = 8
                classified_array[(wsi_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_wsi(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Crop Yield
class CropYieldIndexView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return { 
                    input: ["B04", "B08", "B02", "B11", "SCL"],  // Added B11 for better vegetation detection under trees
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Exclude clouds and other invalid pixels, keep tree canopy (SCL = 4) for processing
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude clouds, cloud shadows, and water
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB08 = [], validValuesB04 = [], validValuesB02 = [], validValuesB11 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B04 > 0 && sample.B02 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB08[a] = sample.B08;
                            validValuesB04[a] = sample.B04;
                            validValuesB02[a] = sample.B02;
                            validValuesB11[a] = sample.B11;  // Using SWIR for under-canopy plants
                            a++;
                        }
                    }
                }

                var arvi;
                if (a > 0) {
                    var NIR = getValue(validValuesB08);
                    var RED = getValue(validValuesB04);
                    var BLUE = getValue(validValuesB02);
                    var avgB11 = getValue(validValuesB11);

                    // Adjusting ARVI for plants under tree canopies using B11
                    if (avgB11 > 0.3) { // Threshold to identify tree canopies
                        arvi = (NIR - (2 * RED - BLUE)) / (NIR + (2 * RED - BLUE + avgB11));
                    } else {
                        arvi = (NIR - (2 * RED - BLUE)) / (NIR + (2 * RED - BLUE));
                    }
                } else {
                    arvi = -9999; // No valid data
                }

                return [arvi];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            # Check if the response is empty (all invalid values)
            if np.all(response == -9999):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_cyi(cyi_array):
                classified_array = np.zeros_like(cyi_array, dtype=np.uint8)
                classified_array[(cyi_array <= 0) & (cyi_array != -9999)] = 1
                classified_array[(cyi_array > 0) & (cyi_array <= 0.1)] = 2
                classified_array[(cyi_array > 0.1) & (cyi_array <= 0.2)] = 3
                classified_array[(cyi_array > 0.2) & (cyi_array <= 0.3)] = 4
                classified_array[(cyi_array > 0.3) & (cyi_array <= 0.4)] = 5
                classified_array[(cyi_array > 0.4) & (cyi_array <= 0.5)] = 6
                classified_array[(cyi_array > 0.5) & (cyi_array <= 0.6)] = 7
                classified_array[(cyi_array > 0.6) & (cyi_array <= 1)] = 8
                classified_array[(cyi_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_cyi(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Disease Weed
class ARVIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return {
                    input: ["B02", "B04", "B08", "B11", "SCL"],  // Include Blue, Red, NIR, SWIR bands and Scene Classification Layer (SCL)
                    output: {
                        id: "default",
                        bands: 1,
                        sampleType: "FLOAT32"
                    },
                    mosaicking: "ORBIT"  // Use ORBIT-based mosaicking
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);  // Filter scenes from the last 3 months
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);  // Use first quartile as the value
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Exclude cloud and other invalid pixels, keep tree canopy (SCL = 4) for processing
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false;  // Exclude clouds, cloud shadows, and water
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB08 = [], validValuesB04 = [], validValuesB02 = [], validValuesB11 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B04 > 0 && sample.B02 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB08[a] = sample.B08;
                            validValuesB04[a] = sample.B04;
                            validValuesB02[a] = sample.B02;  
                            validValuesB11[a] = sample.B11;  // Using SWIR for under-canopy plants
                            a++;
                        }
                    }
                }

                var ARVI;
                if (a > 0) {
                    var avgB08 = getValue(validValuesB08);
                    var avgB04 = getValue(validValuesB04);
                    var avgB02 = getValue(validValuesB02);
                    var avgB11 = getValue(validValuesB11);
                    
                    // Adjust ARVI calculation to account for plants under tree canopies
                    if (avgB11 > 0.3) { // Threshold to identify tree canopies
                        // Adjust ARVI for under-canopy plants using SWIR
                        ARVI = (avgB08 - (2 * avgB04 - avgB02)) / (avgB08 + (2 * avgB04 - avgB02) + avgB11);
                    } else {
                        // Regular ARVI calculation
                        ARVI = (avgB08 - (2 * avgB04 - avgB02)) / (avgB08 + (2 * avgB04 - avgB02));
                    }
                } else {
                    ARVI = -9999;  // No valid data
                }

                return [ARVI];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            if np.all(response == 0):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            # The reclassification logic is the same as before
            def reclassify_arvi(arvi_array):
                classified_array = np.zeros_like(arvi_array, dtype=np.uint8)
                classified_array[(arvi_array <= 0) & (arvi_array != -9999)] = 1
                classified_array[(arvi_array > 0) & (arvi_array <= 0.1)] = 2
                classified_array[(arvi_array > 0.1) & (arvi_array <= 0.2)] = 3
                classified_array[(arvi_array > 0.2) & (arvi_array <= 0.3)] = 4
                classified_array[(arvi_array > 0.3) & (arvi_array <= 0.4)] = 5
                classified_array[(arvi_array > 0.4) & (arvi_array <= 0.5)] = 6
                classified_array[(arvi_array > 0.5) & (arvi_array <= 0.6)] = 7
                classified_array[(arvi_array > 0.6) & (arvi_array <= 1)] = 8
                classified_array[(arvi_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_arvi(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Chlorophyll
class CARIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return {
                    input: ["B03", "B04", "B08", "B11", "SCL"], // Added B11 for better vegetation detection
                    output: {
                        id: "default",
                        bands: 1,
                        sampleType: "FLOAT32"
                    },
                    mosaicking: "ORBIT"
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000); // 3-month filter
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Exclude clouds, cloud shadows, and water
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false;
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB08 = [], validValuesB04 = [], validValuesB03 = [], validValuesB11 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B04 > 0 && sample.B03 > 0 && sample.B11 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB08[a] = sample.B08;
                            validValuesB04[a] = sample.B04;
                            validValuesB03[a] = sample.B03;
                            validValuesB11[a] = sample.B11; // Using B11 for vegetation under trees
                            a++;
                        }
                    }
                }

                var CARI;
                if (a > 0) {
                    var GREEN = getValue(validValuesB03);
                    var RED = getValue(validValuesB04);
                    var NIR = getValue(validValuesB08);
                    var SWIR = getValue(validValuesB11); // Incorporating B11

                    // Calculate CARI considering vegetation under trees
                    var term1 = Math.pow((NIR - GREEN) / 150, 2);
                    var term2 = Math.pow((RED - GREEN), 2);
                    CARI = Math.sqrt(term1 + term2);

                    // Adjust CARI calculation for conditions under tree canopies
                    if (SWIR > 0.3) { // Threshold for identifying vegetation under canopies
                        CARI *= 1.1; // Example adjustment factor for under-canopy areas
                    }
                } else {
                    CARI = -9999; // No valid data
                }

                return [CARI];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                ],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            if np.all(response == -9999):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_cari(cari_array):
                classified_array = np.zeros_like(cari_array, dtype=np.uint8)
                classified_array[(cari_array <= 0) & (cari_array != -9999)] = 1
                classified_array[(cari_array > 0) & (cari_array <= 0.1)] = 2
                classified_array[(cari_array > 0.1) & (cari_array <= 0.2)] = 3
                classified_array[(cari_array > 0.2) & (cari_array <= 0.3)] = 4
                classified_array[(cari_array > 0.3) & (cari_array <= 0.4)] = 5
                classified_array[(cari_array > 0.4) & (cari_array <= 0.5)] = 6
                classified_array[(cari_array > 0.5) & (cari_array <= 0.6)] = 7
                classified_array[(cari_array > 0.6) & (cari_array <= 1)] = 8
                classified_array[(cari_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_cari(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Chlorophyll Growth
class MCARIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return {
                    input: ["B02", "B03", "B04", "B08", "B11", "SCL"],  // Added B11 for better vegetation detection under trees
                    output: {
                        id: "default",
                        bands: 1,
                        sampleType: "FLOAT32"
                    },
                    mosaicking: "ORBIT"
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000); // Last 3 months of data
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Exclude cloud and other invalid pixels, keep tree canopy (SCL = 4) for processing
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude clouds, cloud shadows, and water
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesBlue = [], validValuesGreen = [], validValuesRed = [], validValuesNIR = [], validValuesSWIR = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B02 > 0 && sample.B03 > 0 && sample.B04 > 0 && sample.B08 > 0 && sample.B11 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesBlue[a] = sample.B02;
                            validValuesGreen[a] = sample.B03;
                            validValuesRed[a] = sample.B04;
                            validValuesNIR[a] = sample.B08; // Including NIR for under-canopy plants
                            validValuesSWIR[a] = sample.B11; // SWIR for canopy detection
                            a++;
                        }
                    }
                }

                var mcari;
                if (a > 0) {
                    // Calculate MCARI using valid values
                    var BLUE = getValue(validValuesBlue);
                    var GREEN = getValue(validValuesGreen);
                    var RED = getValue(validValuesRed);
                    var NIR = getValue(validValuesNIR);
                    var SWIR = getValue(validValuesSWIR);

                    mcari = (RED - GREEN) - 0.2 * (RED - BLUE) * (RED / NIR);

                    // Canopy detection adjustment (using SWIR)
                    if (SWIR > 0.3) { // SWIR threshold to identify canopy
                        mcari *= 1.1; // Adjust MCARI value for areas under tree canopy
                    }
                } else {
                    mcari = -9999; // No valid data
                }

                return [mcari];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            if np.all(response == -9999):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_mcari(mcari_array):
                classified_array = np.zeros_like(mcari_array, dtype=np.uint8)
                classified_array[(mcari_array <= 0) & (mcari_array != -9999)] = 1
                classified_array[(mcari_array > 0) & (mcari_array <= 0.1)] = 2
                classified_array[(mcari_array > 0.1) & (mcari_array <= 0.2)] = 3
                classified_array[(mcari_array > 0.2) & (mcari_array <= 0.3)] = 4
                classified_array[(mcari_array > 0.3) & (mcari_array <= 0.4)] = 5
                classified_array[(mcari_array > 0.4) & (mcari_array <= 0.5)] = 6
                classified_array[(mcari_array > 0.5) & (mcari_array <= 0.6)] = 7
                classified_array[(mcari_array > 0.6) & (mcari_array <= 1)] = 8
                classified_array[(mcari_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_mcari(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#Disease Biomass
class TVIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        date = request.data.get('date')
        if not date:
            return Response({'error': 'Date is required.'}, status=status.HTTP_400_BAD_REQUEST)

        evalscript = """
        function setup() {
            return {
                input: ["B08", "B04", "B03", "SCL"], // NIR, Red, Green, Scene Classification Layer
                output: { bands: 2, sampleType: "FLOAT32" }, 
                mosaicking: "ORBIT"
            };
        }

        function validate(sample) {
            var scl = sample.SCL;
            if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                return false; // Exclude clouds, shadows, and invalid pixels
            }
            return true;
        }

        function evaluatePixel(samples) {
            var validValuesB08 = [], validValuesB03 = [], validValuesB04 = [];
            var a = 0;

            for (var i = 0; i < samples.length; i++) {
                var sample = samples[i];
                if (sample.B08 > 0 && sample.B04 > 0 && sample.B03 > 0) {
                    var isValid = validate(sample);
                    if (isValid) {
                        validValuesB08[a] = sample.B08;  // NIR
                        validValuesB04[a] = sample.B04;  // Red
                        validValuesB03[a] = sample.B03;  // Green
                        a++;
                    }
                }
            }

            var tviValue = -9999; // Default invalid value
            var canopyFlag = 0;   // Default flag for canopy detection
            if (a > 0) {
                var avgB08 = validValuesB08.reduce((sum, val) => sum + val, 0) / a;
                var avgB03 = validValuesB03.reduce((sum, val) => sum + val, 0) / a;
                var avgB04 = validValuesB04.reduce((sum, val) => sum + val, 0) / a;

                tviValue = 0.5 * (120 * (avgB08 - avgB03) - 200 * (avgB04 - avgB03)); // Calculate TVI
                if (avgB08 > 0.3) { // Canopy threshold example
                    canopyFlag = 1;
                }
            }

            return [tviValue, canopyFlag];
        }
        """

        sentinel_request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
            responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
            bbox=bbox,
            size=[512, 354.253],
            config=config,
        )

        response = sentinel_request.get_data()[0]

        if np.all(response == -9999):
            return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

        transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

        def reclassify_tvi(data_array):
            tvi_array, canopy_array = data_array[..., 0], data_array[..., 1]
            classified_array = np.zeros_like(tvi_array, dtype=np.uint8)
            
            # Reclassify TVI values
            classified_array[(tvi_array <= -0.3) & (tvi_array != -9999)] = 1  # Low biomass
            classified_array[(tvi_array > -0.3) & (tvi_array <= 0.1)] = 2  # Moderate biomass
            classified_array[(tvi_array > 0.1)] = 3  # High biomass
            classified_array[(canopy_array == 1)] = 4  # Canopy
            classified_array[(tvi_array == -9999)] = 0  # Invalid pixels
            return classified_array

        classified_image = reclassify_tvi(response)
        shapes_gen = shapes(classified_image, mask=None, transform=transform)
        geometries = list(shapes_gen)

        features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
        geojson_data = {"type": "FeatureCollection", "features": features}

        geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
        geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
        intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
        intersection_geojson = intersection_df.to_json()

        return JsonResponse(json.loads(intersection_geojson))


#Deforestration
class DeforestationView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        date = request.data.get('date')
        if not date:
            return Response({'error': 'Date is required.'}, status=status.HTTP_400_BAD_REQUEST)

        evalscript = """
        function setup() {
            return {
                input: ["B08", "B04", "SCL"],  // B08 for NIR, B04 for Red, SCL for scene classification
                output: { bands: 2, sampleType: "FLOAT32" },
                mosaicking: "ORBIT"
            };
        }

        function validate(sample) {
            var scl = sample.SCL;
            // Exclude cloud and other invalid pixels
            if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                return false;
            }
            return true;
        }

        function evaluatePixel(samples) {
            var validValuesB08 = [], validValuesB04 = [];
            var a = 0;

            for (var i = 0; i < samples.length; i++) {
                var sample = samples[i];
                if (sample.B08 > 0 && sample.B04 > 0) {
                    var isValid = validate(sample);
                    if (isValid) {
                        validValuesB08[a] = sample.B08;  // NIR
                        validValuesB04[a] = sample.B04;  // Red
                        a++;
                    }
                }
            }

            var ndvi = -9999; // Default value
            if (a > 0) {
                // Calculate NDVI
                var avgNIR = validValuesB08.reduce((a, b) => a + b, 0) / a;
                var avgRed = validValuesB04.reduce((a, b) => a + b, 0) / a;
                ndvi = (avgNIR - avgRed) / (avgNIR + avgRed);
            }

            // Classification logic: NDVI-based forest and deforestation detection
            var class_no = 0;  // Default class
            if (ndvi >= 0.5) {
                class_no = 2;  // Forested
            } else if (ndvi >= 0.2) {
                class_no = 1;  // Deforested
            }

            return [ndvi, class_no];
        }
        """

        sentinel_request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
            responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
            bbox=bbox,
            size=[512, 354.253],
            config=config,
        )

        response = sentinel_request.get_data()[0]

        # Check if the response contains valid data
        if np.all(response == -9999):
            return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

        transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

        def reclassify_deforestation(data_array):
            ndvi_array, class_array = data_array[..., 0], data_array[..., 1]
            classified_array = np.zeros_like(ndvi_array, dtype=np.uint8)

            # Classify NDVI values for deforestation detection
            classified_array[(class_array == 1)] = 1  # Deforested
            classified_array[(class_array == 2)] = 2  # Forested
            classified_array[(ndvi_array == -9999)] = 0  # Invalid pixels
            return classified_array

        classified_image = reclassify_deforestation(response)
        shapes_gen = shapes(classified_image, mask=None, transform=transform)
        geometries = list(shapes_gen)

        features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
        geojson_data = {"type": "FeatureCollection", "features": features}

        geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
        geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
        intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
        intersection_geojson = intersection_df.to_json()

        return JsonResponse(json.loads(intersection_geojson))


# Disease Detection (EVI)
class EVIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        date = request.data.get('date')
        if not date:
            return Response({'error': 'Date is required.'}, status=status.HTTP_400_BAD_REQUEST)

        evalscript = """
        function setup() {
            return {
                input: ["B08", "B04", "B02", "SCL"], // NIR, Red, Blue, Scene Classification Layer
                output: { bands: 2, sampleType: "FLOAT32" }, 
                mosaicking: "ORBIT"
            };
        }

        function validate(sample) {
            var scl = sample.SCL;
            if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                return false; // Exclude clouds, shadows, and invalid pixels
            }
            return true;
        }

        function evaluatePixel(samples) {
            var validValuesB08 = [], validValuesB04 = [], validValuesB02 = [];
            var a = 0;

            for (var i = 0; i < samples.length; i++) {
                var sample = samples[i];
                if (sample.B08 > 0 && sample.B04 > 0 && sample.B02 > 0) {
                    var isValid = validate(sample);
                    if (isValid) {
                        validValuesB08[a] = sample.B08;  // NIR
                        validValuesB04[a] = sample.B04;  // Red
                        validValuesB02[a] = sample.B02;  // Blue
                        a++;
                    }
                }
            }

            var eviValue = -9999; // Default invalid value
            var diseaseFlag = 0;   // Default flag for disease detection
            if (a > 0) {
                var avgB08 = validValuesB08.reduce((sum, val) => sum + val, 0) / a;
                var avgB04 = validValuesB04.reduce((sum, val) => sum + val, 0) / a;
                var avgB02 = validValuesB02.reduce((sum, val) => sum + val, 0) / a;

                // Calculate EVI
                eviValue = 2.5 * ((avgB08 - avgB04) / (avgB08 + 6 * avgB04 - 7.5 * avgB02 + 1));
                if (eviValue < 0.2) { // Example threshold for disease detection
                    diseaseFlag = 1;
                }
            }

            return [eviValue, diseaseFlag];
        }
        """

        sentinel_request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
            responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
            bbox=bbox,
            size=[512, 354.253],
            config=config,
        )

        response = sentinel_request.get_data()[0]

        if np.all(response == -9999):
            return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

        transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

        def reclassify_evi(data_array):
            evi_array, disease_array = data_array[..., 0], data_array[..., 1]
            classified_array = np.zeros_like(evi_array, dtype=np.uint8)
            
            # Reclassify EVI values
            classified_array[(evi_array < 0.2) & (evi_array != -9999)] = 1  # Potential disease presence
            classified_array[(evi_array >= 0.2) & (evi_array <= 0.5)] = 2  # Moderate vegetation
            classified_array[(evi_array > 0.5)] = 3  # Healthy vegetation
            classified_array[(disease_array == 1)] = 4  # Disease flag
            classified_array[(evi_array == -9999)] = 0  # Invalid pixels
            return classified_array

        classified_image = reclassify_evi(response)
        shapes_gen = shapes(classified_image, mask=None, transform=transform)
        geometries = list(shapes_gen)

        features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
        geojson_data = {"type": "FeatureCollection", "features": features}

        geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
        geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
        intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
        intersection_geojson = intersection_df.to_json()

        return JsonResponse(json.loads(intersection_geojson))



######################## FORECAST ENDPOINTS BELOW #####################



# Soil Quality Forecast
class SoilQualityFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)

            evalscript = """
            function setup() {
                return { 
                    input: ["B02", "B03", "B04", "B08", "SCL"], 
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function validate(sample) {
                var scl = sample.SCL;
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false;
                }
                return true;
            }

            function evaluatePixel(samples) {
                var validB02 = [];
                var validB03 = [];
                var validB04 = [];
                var validB08 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B02 > 0 && sample.B03 > 0 && sample.B04 > 0 && sample.B08 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validB02[a] = sample.B02;
                            validB03[a] = sample.B03;
                            validB04[a] = sample.B04;
                            validB08[a] = sample.B08;
                            a++;
                        }
                    }
                }

                var sqi;
                if (a > 0) {
                    var avgB02 = validB02.reduce((a, b) => a + b, 0) / validB02.length;
                    var avgB03 = validB03.reduce((a, b) => a + b, 0) / validB03.length;
                    var avgB04 = validB04.reduce((a, b) => a + b, 0) / validB04.length;
                    var avgB08 = validB08.reduce((a, b) => a + b, 0) / validB08.length;

                    // Example Soil Quality Index formula
                    sqi = (avgB08 - avgB04) / (avgB08 + avgB04 + avgB02 + avgB03);
                } else {
                    sqi = -9999;
                }

                return [sqi];
            }
            """

            try:
                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A, 
                        time_interval=(date, date)
                    )],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_sqi(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [
                    {"type": "Feature", "geometry": geom, "properties": {"class_no": value}}
                    for geom, value in geometries if value != 0
                ]
                geojson_data = {"type": "FeatureCollection", "features": features}

                results.extend(self.process_geojson_data(geojson_data, polygon))

            except Exception as e:
                return Response({'error': f'Error in SentinelHub request: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"type": "FeatureCollection", "features": results}, status=status.HTTP_200_OK)

    def reclassify_sqi(self, sqi_array):
        classified_array = np.zeros_like(sqi_array, dtype=np.uint8)
        classified_array[(sqi_array <= -0.2) & (sqi_array != -9999)] = 1
        classified_array[(sqi_array > -0.2) & (sqi_array <= 0)] = 2
        classified_array[(sqi_array > 0) & (sqi_array <= 0.2)] = 3
        classified_array[(sqi_array > 0.2) & (sqi_array <= 0.4)] = 4
        classified_array[(sqi_array > 0.4)] = 5
        classified_array[(sqi_array == -9999)] = 0
        return classified_array

    def process_geojson_data(self, geojson_data, polygon):
        geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
        geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
        intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)

        features = []
        for _, row in intersection_df.iterrows():
            features.append({
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": row.drop("geometry").to_dict()
            })

        return features

# Vegetation Health Forecast
class NDVIFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)

            evalscript = """
            function setup() {
                return { 
                    input: ["B04", "B08", "SCL"], 
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function validate(sample) {
                var scl = sample.SCL;
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false;
                }
                return true;
            }

            function evaluatePixel(samples) {
                var validB08 = [];
                var validB04 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B04 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validB08[a] = sample.B08;
                            validB04[a] = sample.B04;
                            a++;
                        }
                    }
                }

                var ndvi;
                if (a > 0) {
                    var avgB08 = validB08.reduce((a, b) => a + b, 0) / validB08.length;
                    var avgB04 = validB04.reduce((a, b) => a + b, 0) / validB04.length;

                    ndvi = (avgB08 - avgB04) / (avgB08 + avgB04);
                } else {
                    ndvi = -9999;
                }

                return [ndvi];
            }
            """

            try:
                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A, 
                        time_interval=(date, date)
                    )],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_ndvi(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [
                    {"type": "Feature", "geometry": geom, "properties": {"class_no": value}}
                    for geom, value in geometries if value != 0
                ]
                geojson_data = {"type": "FeatureCollection", "features": features}

                results.extend(self.process_geojson_data(geojson_data, polygon))

            except Exception as e:
                return Response({'error': f'Error in SentinelHub request: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"type": "FeatureCollection", "features": results}, status=status.HTTP_200_OK)

    def reclassify_ndvi(self, ndvi_array):
        classified_array = np.zeros_like(ndvi_array, dtype=np.uint8)
        classified_array[(ndvi_array <= 0) & (ndvi_array != -9999)] = 1
        classified_array[(ndvi_array > 0) & (ndvi_array <= 0.1)] = 2
        classified_array[(ndvi_array > 0.1) & (ndvi_array <= 0.2)] = 3
        classified_array[(ndvi_array > 0.2) & (ndvi_array <= 0.4)] = 4
        classified_array[(ndvi_array > 0.4) & (ndvi_array <= 0.5)] = 5
        classified_array[(ndvi_array > 0.5) & (ndvi_array <= 0.6)] = 6
        classified_array[(ndvi_array > 0.6) & (ndvi_array <= 0.7)] = 7
        classified_array[(ndvi_array > 0.7)] = 8
        classified_array[(ndvi_array == -9999)] = 0
        return classified_array

    def process_geojson_data(self, geojson_data, polygon):
        geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
        geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
        intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)

        features = []
        for _, row in intersection_df.iterrows():
            features.append({
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": row.drop("geometry").to_dict()
            })

        return features


# Humidity Forecast
class NDWIFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []
        
        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception as e:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
            serializer = IndicesSerializer(data={'date': date})  # Adjust data as needed
            if serializer.is_valid():
                evalscript = """
                function setup() {
                    return { 
                        input: ["B03", "B08", "B11", "SCL"],  // Added B11 for detecting vegetation under trees
                        output: { bands: 1, sampleType: "FLOAT32" }, 
                        mosaicking: "ORBIT" 
                    };
                }

                function preProcessScenes(collections) {
                    collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                        var orbitDateFrom = new Date(orbit.dateFrom);
                        return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                    });
                    return collections;
                }

                function getValue(values) {
                    values.sort(function (a, b) { return a - b; });
                    return getFirstQuartile(values);
                }

                function getFirstQuartile(sortedValues) {
                    var index = Math.floor(sortedValues.length / 4);
                    return sortedValues[index];
                }

                function validate(sample) {
                    var scl = sample.SCL;
                    // Using SCL to filter out clouds, shadows, and invalid pixels
                    if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                        return false; // Exclude cloud and cloud shadow pixels
                    }
                    return true;
                }

                function evaluatePixel(samples, scenes) {
                    var validValuesB03 = [], validValuesB08 = [], validValuesB11 = [];
                    var a = 0;

                    for (var i = 0; i < samples.length; i++) {
                        var sample = samples[i];
                        if (sample.B03 > 0 && sample.B08 > 0) {
                            var isValid = validate(sample);
                            if (isValid) {
                                validValuesB03[a] = sample.B03;
                                validValuesB08[a] = sample.B08;
                                validValuesB11[a] = sample.B11;  // B11 added for under-tree vegetation detection
                                a++;
                            }
                        }
                    }

                    var ndwi;
                    if (a > 0) {
                        var avgB03 = getValue(validValuesB03);
                        var avgB08 = getValue(validValuesB08);
                        var avgB11 = getValue(validValuesB11);

                        if (avgB11 > 0.3) {  // Threshold to account for vegetation under tree canopies
                            // Adjust NDWI considering B11 for under-canopy water
                            ndwi = (avgB03 - avgB08) / (avgB03 + avgB08 + avgB11);
                        } else {
                            // Standard NDWI calculation
                            ndwi = (avgB03 - avgB08) / (avgB03 + avgB08);
                        }
                    } else {
                        ndwi = -9999; // No valid data
                    }

                    return [ndwi];
                }
                """

                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[
                        SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                    ],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354.253],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue 

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_ndwi(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
                geojson_data = {"type": "FeatureCollection", "features": features}

                geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
                geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
                intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
                intersection_geojson = intersection_df.to_json()

                results.append(json.loads(intersection_geojson))

        predicted_results = self.predict_ndwi(results)
        return Response(predicted_results, status=status.HTTP_200_OK)

    def reclassify_ndwi(self, ndwi_array):
        classified_array = np.zeros_like(ndwi_array, dtype=np.uint8)
        classified_array[(ndwi_array <= 0) & (ndwi_array != -9999)] = 1
        classified_array[(ndwi_array > 0) & (ndwi_array <= 0.1)] = 2
        classified_array[(ndwi_array > 0.1) & (ndwi_array <= 0.2)] = 3
        classified_array[(ndwi_array > 0.2) & (ndwi_array <= 0.4)] = 4
        classified_array[(ndwi_array > 0.4) & (ndwi_array <= 0.5)] = 5
        classified_array[(ndwi_array > 0.5) & (ndwi_array <= 0.6)] = 6
        classified_array[(ndwi_array > 0.6) & (ndwi_array <= 0.7)] = 7
        classified_array[(ndwi_array > 0.7) & (ndwi_array <= 1)] = 8
        classified_array[(ndwi_array == -9999)] = 0  # Set cloudy pixels to 0
        return classified_array

    def predict_ndwi(self, results):
        predicted_features = []
        valid_coordinates = []
        valid_class_numbers = []
        valid_results = []

        if not results or not isinstance(results, list):
            print("No results provided or results are not in the expected format.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features 
            }

        for feature in results:
            if 'features' not in feature:
                print("Feature missing 'features' key:", feature)
                continue

            for item in feature['features']:
                geometry = item.get('geometry', {})
                coords = geometry.get('coordinates', [])

                if isinstance(coords, list) and coords and isinstance(coords[0], list):
                    if len(coords[0]) > 0 and len(coords[0][0]) == 2:
                        valid_coord = coords[0][0]
                        valid_coordinates.append(valid_coord)
                        valid_results.append(item)

                        class_no = item.get('properties', {}).get('class_no')
                        if class_no is not None:
                            valid_class_numbers.append(class_no)
                        else:
                            print("Class number missing in properties.")
                    else:
                        a = 3 #print(f"Invalid coordinate structure found: {coords}")
                else:
                    print(f"Invalid coordinate found: {coords}")

        if not valid_coordinates or not valid_class_numbers:
            print("No valid coordinates or class numbers found.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features
            }

        coordinates_array = np.array(valid_coordinates)
        class_numbers_array = np.array(valid_class_numbers)

        # Placeholder for prediction logic, you can implement your own model prediction
        predicted_class = np.round(np.mean(class_numbers_array)).astype(int) 

        # Formulate output features
        for idx, geometry in enumerate(valid_results):
            predicted_features.append({
                "type": "Feature",
                "geometry": geometry['geometry'],
                "properties": {"predicted_class": predicted_class}
            })

        return {
            "type": "FeatureCollection",
            "features": predicted_features
        }


# Plant Moisture Forecast
class NDMIFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []
        
        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception as e:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)

            cloud_coverage = int(request.data.get('cloud_coverage', 20))
            if cloud_coverage > 100:
                return Response({'error': 'Cloud coverage value cannot be greater than 100.'}, status=status.HTTP_400_BAD_REQUEST)

            evalscript = """
            //VERSION=3
            function setup() {
                return { input: ["B08", "B11", "CLM"], output: { bands: 1, sampleType: "FLOAT32" } };
            }
            function evaluatePixel(sample) {
                if (sample.CLM == 1) {
                    return [-9999];
                }
                return [(sample.B08 - sample.B11) / (sample.B08 + sample.B11)];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                ],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            if np.all(response == -9999):
                continue 

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            classified_image = self.reclassify_ndmi(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            results.append(json.loads(intersection_geojson))

        predicted_results = self.predict_ndmi(results)
        return Response(predicted_results, status=status.HTTP_200_OK)

    def reclassify_ndmi(self, ndmi_array):
        classified_array = np.zeros_like(ndmi_array, dtype=np.uint8)
        classified_array[(ndmi_array <= -1) & (ndmi_array != -9999)] = 1
        classified_array[(ndmi_array > -1) & (ndmi_array <= 0)] = 2
        classified_array[(ndmi_array > 0) & (ndmi_array <= 0.1)] = 3
        classified_array[(ndmi_array > 0.1) & (ndmi_array <= 0.2)] = 4
        classified_array[(ndmi_array > 0.2) & (ndmi_array <= 0.3)] = 5
        classified_array[(ndmi_array > 0.3) & (ndmi_array <= 0.4)] = 6
        classified_array[(ndmi_array > 0.4) & (ndmi_array <= 0.5)] = 7
        classified_array[(ndmi_array > 0.5) & (ndmi_array <= 1)] = 8
        classified_array[(ndmi_array == -9999)] = 0  # Set cloudy pixels to 0
        return classified_array

    def predict_ndmi(self, results):
        predicted_features = []
        valid_coordinates = []
        valid_class_numbers = []
        valid_results = []

        if not results or not isinstance(results, list):
            print("No results provided or results are not in the expected format.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features 
            }

        for feature in results:
            if 'features' not in feature:
                print("Feature missing 'features' key:", feature)
                continue

            for item in feature['features']:
                geometry = item.get('geometry', {})
                coords = geometry.get('coordinates', [])

                if isinstance(coords, list) and coords and isinstance(coords[0], list):
                    if len(coords[0]) > 0 and len(coords[0][0]) == 2:
                        valid_coord = coords[0][0]
                        valid_coordinates.append(valid_coord)
                        valid_results.append(item)

                        class_no = item.get('properties', {}).get('class_no')
                        if class_no is not None:
                            valid_class_numbers.append(class_no)
                        else:
                            print("Class number missing in properties.")
                    else:
                        a = 3 #print(f"Invalid coordinate structure found: {coords}")
                else:
                    print(f"Invalid coordinate found: {coords}")

        if not valid_coordinates or not valid_class_numbers:
            print("No valid coordinates or class numbers found.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features
            }

        coordinates_array = np.array(valid_coordinates)
        class_numbers_array = np.array(valid_class_numbers)

        model = LinearRegression()
        model.fit(coordinates_array, class_numbers_array) 

        max_sample_size = min(1000, len(coordinates_array))
        sampled_indices = random.sample(range(len(coordinates_array)), max_sample_size)

        sampled_coordinates = coordinates_array[sampled_indices]
        predicted_class_no = model.predict(sampled_coordinates).astype(int)

        predicted_features = [
            {
                "id": str(i),
                "type": "Feature",
                "properties": {
                    "class_no": class_no
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": valid_results[idx]['geometry']['coordinates'] 
                }
            }
            for i, (idx, class_no) in enumerate(zip(sampled_indices, predicted_class_no))
        ]

        predicted_geojson = {
            "type": "FeatureCollection",
            "features": predicted_features
        }

        return predicted_geojson


# Coffee Ripeness Forecast
class CRIFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception as e:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
            serializer = IndicesSerializer(data={'date': date})
            if serializer.is_valid():
                evalscript = """
                function setup() {
                    return { 
                        input: ["B04", "B11", "SCL"],  // Included B11 for detecting vegetation under trees
                        output: { bands: 1, sampleType: "FLOAT32" }, 
                        mosaicking: "ORBIT" 
                    };
                }

                function preProcessScenes(collections) {
                    collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                        var orbitDateFrom = new Date(orbit.dateFrom);
                        return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                    });
                    return collections;
                }

                function getValue(values) {
                    values.sort(function (a, b) { return a - b; });
                    return getFirstQuartile(values);
                }

                function getFirstQuartile(sortedValues) {
                    var index = Math.floor(sortedValues.length / 4);
                    return sortedValues[index];
                }

                function validate(sample) {
                    var scl = sample.SCL;
                    // Exclude cloud and other invalid pixels, keep tree canopy (SCL = 4) for processing
                    if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                        return false; // Exclude clouds, cloud shadows, and water
                    }
                    return true;
                }

                function evaluatePixel(samples, scenes) {
                    var validValuesB04 = [];
                    var validValuesB11 = [];  // For Canopy detection using B11 (SWIR)
                    var a = 0;

                    for (var i = 0; i < samples.length; i++) {
                        var sample = samples[i];
                        if (sample.B04 > 0) {
                            var isValid = validate(sample);
                            if (isValid) {
                                validValuesB04[a] = sample.B04;
                                validValuesB11[a] = sample.B11; // Capture B11 (SWIR) for canopy
                                a++;
                            }
                        }
                    }

                    var cri;
                    var canopyIndex;
                    if (a > 0) {
                        cri = getValue(validValuesB04);
                        canopyIndex = getValue(validValuesB11); // Compute canopy index
                    } else {
                        cri = -9999; // No valid data
                        canopyIndex = -9999; // No valid canopy data
                    }

                    return [cri, canopyIndex];  // Return both ripeness and canopy data
                }
                """

                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354.253],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image, canopy_data = self.reclassify_ripeness_and_canopy(response)

                # Continue using reclassified data for ripeness as before
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [{"type": "Feature", "geometry": geom, "properties": {"ripeness_class": value, "canopy_index": canopy_data[idx]}} 
                            for idx, (geom, value) in enumerate(geometries) if value != 0]

                geojson_data = {"type": "FeatureCollection", "features": features}

                geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
                geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
                intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
                intersection_geojson = intersection_df.to_json()

                results.append(json.loads(intersection_geojson))

        predicted_results = self.predict_ripeness(results)
        return Response(predicted_results, status=status.HTTP_200_OK)

    def reclassify_ripeness_and_canopy(self, ripeness_array):
        # Reclassify ripeness and calculate canopy index
        classified_array = np.zeros_like(ripeness_array, dtype=np.uint8)
        canopy_index_array = np.zeros_like(ripeness_array, dtype=np.float32)

        classified_array[(ripeness_array <= 0) & (ripeness_array != -9999)] = 1  # Low ripeness
        classified_array[(ripeness_array > 0) & (ripeness_array <= 0.3)] = 2  # Medium ripeness
        classified_array[(ripeness_array > 0.3)] = 3  # High ripeness
        classified_array[(ripeness_array == -9999)] = 0  # Cloudy pixels

        # For canopy, we could classify based on B11 (SWIR) values
        # Canopy can be classified based on SWIR (B11) index values
        canopy_index_array[(ripeness_array != -9999)] = ripeness_array  # Using the same array for simplicity

        return classified_array, canopy_index_array

    def predict_ripeness(self, results):
        # Same as the original function, but now we include canopy_index in the results
        predicted_features = []
        valid_coordinates = []
        valid_ripeness_classes = []
        valid_canopy_indexes = []
        valid_results = []

        if not results or not isinstance(results, list):
            return {
                "type": "FeatureCollection",
                "features": predicted_features
            }

        for feature in results:
            if 'features' not in feature:
                continue

            for item in feature['features']:
                geometry = item.get('geometry', {})
                coords = geometry.get('coordinates', [])

                if isinstance(coords, list) and coords and isinstance(coords[0], list):
                    valid_coord = coords[0][0]
                    valid_coordinates.append(valid_coord)
                    valid_results.append(item)

                    ripeness_class = item.get('properties', {}).get('ripeness_class')
                    canopy_index = item.get('properties', {}).get('canopy_index')

                    if ripeness_class is not None:
                        valid_ripeness_classes.append(ripeness_class)
                    if canopy_index is not None:
                        valid_canopy_indexes.append(canopy_index)

        coordinates_array = np.array(valid_coordinates)
        ripeness_classes_array = np.array(valid_ripeness_classes)
        canopy_indexes_array = np.array(valid_canopy_indexes)

        model = LinearRegression()
        model.fit(coordinates_array, ripeness_classes_array)

        sampled_indices = random.sample(range(len(coordinates_array)), min(1000, len(coordinates_array)))

        sampled_coordinates = coordinates_array[sampled_indices]
        predicted_ripeness_class = model.predict(sampled_coordinates).astype(int)

        predicted_features = [
            {
                "id": str(i),
                "type": "Feature",
                "properties": {
                    "ripeness_class": ripeness_class,
                    "canopy_index": canopy_index
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": valid_results[idx]['geometry']['coordinates']
                }
            }
            for i, (idx, ripeness_class, canopy_index) in enumerate(zip(sampled_indices, predicted_ripeness_class, canopy_indexes_array[sampled_indices]))
        ]

        predicted_geojson = {
            "type": "FeatureCollection",
            "features": predicted_features
        }

        return predicted_geojson


# Ground Temperature Forecast
class LSTFView(APIView):
    def post(self, request, format=None):
        geojson_features = request.data.get('features')
        if not geojson_features:
            return Response({'error': 'GeoJSON features are required.'}, status=status.HTTP_400_BAD_REQUEST)

        all_temperatures = []

        # Process each polygon feature in the input
        for feature in geojson_features:
            geojson_polygon = feature.get('geometry')
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                return Response({'error': 'Each feature must have a geometry and a date.'}, status=status.HTTP_400_BAD_REQUEST)

            try:
                polygon = shape(geojson_polygon)
            except Exception as e:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            # Define bounding box based on the polygon
            bbox = polygon.bounds  # (minx, miny, maxx, maxy)
            bbox = BBox(bbox=(bbox[0], bbox[1], bbox[2], bbox[3]), crs=CRS.WGS84)

            # Create and send SentinelHubRequest
            evalscript = """
            //VERSION=3
            var option = 0;
            var minC = 0;
            var maxC = 50;
            var NDVIs = 0.2;
            var NDVIv = 0.8;
            var waterE = 0.991;
            var soilE = 0.966;
            var vegetationE = 0.973;
            var C = 0.009;
            var bCent = 0.000010854;
            var rho = 0.01438;
            let viz = ColorGradientVisualizer.createRedTemperature(minC, maxC);

            function setup() {
                return {
                    input: [
                        { datasource: "S3SLSTR", bands: ["S8"] },
                        { datasource: "S3OLCI", bands: ["B06", "B08", "B17"] }
                    ],
                    output: [
                        { id: "default", bands: 3, sampleType: SampleType.AUTO }
                    ],
                    mosaicking: "ORBIT"
                }
            }

            function LSEcalc(NDVI, Pv) {
                var LSE;
                if (NDVI < 0) {
                    LSE = waterE;
                } else if (NDVI < NDVIs) {
                    LSE = soilE;
                } else if (NDVI > NDVIv) {
                    LSE = vegetationE;
                } else {
                    LSE = vegetationE * Pv + soilE * (1 - Pv) + C;
                }
                return LSE;
            }

            function evaluatePixel(samples) {
                var LSTmax = -999;
                var LSTavg = 0;
                var N = samples.S3SLSTR.length;
                var LSTarray = [];

                for (let i = 0; i < N; i++) {
                    var Bi = samples.S3SLSTR[i].S8;
                    if (Bi <= 173 || Bi >= 65000) continue;

                    var S8BTi = Bi - 273.15;
                    var NDVIi = (samples.S3OLCI[i].B17 - samples.S3OLCI[i].B08) / (samples.S3OLCI[i].B17 + samples.S3OLCI[i].B08);
                    var PVi = Math.pow(((NDVIi - NDVIs) / (NDVIv - NDVIs)), 2);
                    var LSEi = LSEcalc(NDVIi, PVi);
                    var LSTi = (S8BTi / (1 + (((bCent * S8BTi) / rho) * Math.log(LSEi))));

                    LSTavg += LSTi;
                    LSTarray.push(LSTi);
                }

                LSTavg /= N;
                return viz.process(LSTavg);
            }
            """

            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL3_SLSTR,
                        identifier="S3SLSTR",
                        time_interval=(date, date),
                    ),
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL3_OLCI,
                        identifier="S3OLCI",
                        time_interval=(date, date),
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF),
                ],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            # Get the data and calculate temperature
            response = request.get_data()
            if response and len(response) > 0:
                response_data = response[0]
                minC = 0  # Set based on your evalscript
                maxC = 50  # Set based on your evalscript

                response_data_temp = minC + (response_data[..., 0] / 255.0) * (maxC - minC)

                # Calculate mean of response_data_temp and round to nearest integer
                mean_temp = np.round(np.mean(response_data_temp))
                all_temperatures.append(mean_temp)

        # Predict next week's temperature using a simple model
        if len(all_temperatures) > 0:
            predicted_temperature = self.simple_prediction(all_temperatures)
            return Response({'mean_temperature': predicted_temperature}, status=status.HTTP_200_OK)

        return Response({'error': 'No temperature data available for predictions.'}, status=status.HTTP_400_BAD_REQUEST)

    def simple_prediction(self, temperatures):
        # Simple prediction model: average of the collected temperatures
        return int(np.mean(temperatures))  # Return average as the predicted temperature for next week


# Water Stress Forecast
class WaterStressIndexForecastView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        for feature in geojson_collection:
            geojson_polygon = feature.get('geometry')
            date = feature.get('properties', {}).get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)

            evalscript = """
            function setup() {
                return { 
                    input: ["B04", "B08", "B11", "SCL"], 
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function validate(sample) {
                var scl = sample.SCL;
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false;
                }
                return true;
            }

            function evaluatePixel(samples) {
                var validB04 = [];
                var validB08 = [];
                var validB11 = [];
                var count = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (validate(sample) && sample.B04 > 0 && sample.B08 > 0 && sample.B11 > 0) {
                        validB04.push(sample.B04);
                        validB08.push(sample.B08);
                        validB11.push(sample.B11);
                        count++;
                    }
                }

                if (count > 0) {
                    var avgB04 = validB04.reduce((a, b) => a + b, 0) / count;
                    var avgB08 = validB08.reduce((a, b) => a + b, 0) / count;
                    var avgB11 = validB11.reduce((a, b) => a + b, 0) / count;

                    var wst = (avgB04 - avgB08) / (avgB04 + avgB08 + avgB11);
                    return [wst];
                }
                return [-9999];
            }
            """

            try:
                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=(date, date)
                    )],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])
                classified_image = self.reclassify_waterstress(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [
                    {"type": "Feature", "geometry": geom, "properties": {"class_no": value}}
                    for geom, value in geometries if value != 0
                ]
                geojson_data = {"type": "FeatureCollection", "features": features}

                results.extend(self.process_geojson_data(geojson_data, polygon))

            except Exception as e:
                return Response({'error': f'Error in SentinelHub request: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"type": "FeatureCollection", "features": results}, status=status.HTTP_200_OK)

    def reclassify_waterstress(self, wst_array):
        classified_array = np.zeros_like(wst_array, dtype=np.uint8)
        classified_array[(wst_array <= 0) & (wst_array != -9999)] = 1
        classified_array[(wst_array > 0) & (wst_array <= 0.1)] = 2
        classified_array[(wst_array > 0.1) & (wst_array <= 0.2)] = 3
        classified_array[(wst_array > 0.2) & (wst_array <= 0.4)] = 4
        classified_array[(wst_array > 0.4) & (wst_array <= 0.5)] = 5
        classified_array[(wst_array > 0.5) & (wst_array <= 0.6)] = 6
        classified_array[(wst_array > 0.6) & (wst_array <= 0.7)] = 7
        classified_array[(wst_array > 0.7)] = 8
        classified_array[(wst_array == -9999)] = 0
        return classified_array

    def process_geojson_data(self, geojson_data, polygon):
        geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
        geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
        intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)

        features = []
        for _, row in intersection_df.iterrows():
            features.append({
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": row.drop("geometry").to_dict()
            })

        return features


# Crop Yield Forecast
class CropYieldIndexForecastView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)

            evalscript = """
            function setup() {
                return { 
                    input: ["B04", "B08", "B02", "CLM"], 
                    output: { bands: 2, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function evaluatePixel(sample) {
                if (sample.CLM === 1) {
                    return [-9999, 0];
                }
                var NIR = sample.B08;
                var RED = sample.B04;
                var BLUE = sample.B02;

                // Calculate ARVI (Atmospherically Resistant Vegetation Index)
                var ARVI = (NIR - (2 * RED - BLUE)) / (NIR + (2 * RED - BLUE));

                // Canopy detection: Check if ARVI indicates a canopy
                var canopy = ARVI > 0.5 ? 1 : 0;

                return [ARVI, canopy];
            }
            """

            try:
                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A, 
                        time_interval=(date, date)
                    )],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response[..., 0] == -9999):
                    continue

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_arvi(response[..., 0])  # ARVI band
                canopy_mask = response[..., 1]  # Canopy band
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [
                    {
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {
                            "class_no": value,
                            "canopy": canopy_mask[tuple(np.unravel_index(idx, canopy_mask.shape))] if value != 0 else 0
                        },
                    }
                    for idx, (geom, value) in enumerate(geometries) if value != 0
                ]
                geojson_data = {"type": "FeatureCollection", "features": features}

                results.extend(self.process_geojson_data(geojson_data, polygon))

            except Exception as e:
                return Response({'error': f'Error in SentinelHub request: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"type": "FeatureCollection", "features": results}, status=status.HTTP_200_OK)

    def reclassify_arvi(self, arvi_array):
        classified_array = np.zeros_like(arvi_array, dtype=np.uint8)
        classified_array[(arvi_array <= 0) & (arvi_array != -9999)] = 1
        classified_array[(arvi_array > 0) & (arvi_array <= 0.1)] = 2
        classified_array[(arvi_array > 0.1) & (arvi_array <= 0.2)] = 3
        classified_array[(arvi_array > 0.2) & (arvi_array <= 0.4)] = 4
        classified_array[(arvi_array > 0.4) & (arvi_array <= 0.5)] = 5
        classified_array[(arvi_array > 0.5) & (arvi_array <= 0.6)] = 6
        classified_array[(arvi_array > 0.6) & (arvi_array <= 0.7)] = 7
        classified_array[(arvi_array > 0.7)] = 8
        classified_array[(arvi_array == -9999)] = 0
        return classified_array

    def process_geojson_data(self, geojson_data, polygon):
        # Ensure the polygon is valid and converted to GeoDataFrame
        if polygon.is_empty or not polygon.is_valid:
            return []

        geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')

        # Ensure the input geojson_data has valid features
        if not geojson_data['features']:
            return []

        geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data['features'], crs='epsg:4326')

        if 'geometry' not in geojson_data_df.columns or geojson_data_df.geometry.is_empty.any():
            return []

        # Perform the intersection
        try:
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
        except Exception as e:
            raise ValueError(f"Error during GeoDataFrame overlay: {e}")

        features = []
        for _, row in intersection_df.iterrows():
            if row.geometry.is_empty or not row.geometry.is_valid:
                continue
            features.append({
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": row.drop("geometry").to_dict()
            })

        return features


# Disease Weed Forecast
class ARVIFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        for feature in geojson_collection:
            geojson_polygon = feature.get('geometry')
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)

            try:
                cloud_coverage = int(request.data.get('cloud_coverage', 20))
                if cloud_coverage > 100:
                    return Response({'error': 'Cloud coverage value cannot exceed 100.'}, status=status.HTTP_400_BAD_REQUEST)

                evalscript = """
                //VERSION=3
                function setup() {
                  return {
                    input: ["B02", "B04", "B08", "CLM"],
                    output: {
                      id: "default",
                      bands: 2, 
                      sampleType: "FLOAT32"
                    }
                  };
                }

                function evaluatePixel(sample) {
                  if (sample.CLM == 1) {
                    return [-9999, -9999]; // Cloud mask
                  }
                  
                  var NIR = sample.B08;
                  var RED = sample.B04;
                  var BLUE = sample.B02;

                  var ARVI = (NIR - (2 * RED - BLUE)) / (NIR + (2 * RED - BLUE));
                  var NDVI = (NIR - RED) / (NIR + RED);
                  var canopy = NDVI > 0.6 ? 1 : 0;

                  return [ARVI, canopy];
                }
                """

                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=(date, date)
                    )],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354],
                    config=config,
                )

                response = sentinel_request.get_data()[0]
                if np.all(response == -9999):
                    continue

                arvi_band = response[0]
                canopy_band = response[1]
                transform = rasterio.transform.from_bounds(*bbox, response.shape[2], response.shape[1])

                arvi_features = self.process_indices(arvi_band, transform, polygon, self.reclassify_arvi)
                canopy_features = self.process_indices(canopy_band, transform, polygon, self.reclassify_canopy, canopy=True)

                results.extend(arvi_features)
                results.extend(canopy_features)

            except Exception as e:
                return Response({'error': f'Error processing SentinelHub request: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"type": "FeatureCollection", "features": results}, status=status.HTTP_200_OK)

    def reclassify_arvi(self, arvi_array):
        classified_array = np.zeros_like(arvi_array, dtype=np.uint8)
        classified_array[(arvi_array <= 0) & (arvi_array != -9999)] = 1
        classified_array[(arvi_array > 0) & (arvi_array <= 0.1)] = 2
        classified_array[(arvi_array > 0.1) & (arvi_array <= 0.2)] = 3
        classified_array[(arvi_array > 0.2) & (arvi_array <= 0.4)] = 4
        classified_array[(arvi_array > 0.4) & (arvi_array <= 0.5)] = 5
        classified_array[(arvi_array > 0.5) & (arvi_array <= 0.6)] = 6
        classified_array[(arvi_array > 0.6) & (arvi_array <= 0.7)] = 7
        classified_array[(arvi_array > 0.7)] = 8
        classified_array[(arvi_array == -9999)] = 0
        return classified_array

    def reclassify_canopy(self, canopy_array):
        classified_array = np.zeros_like(canopy_array, dtype=np.uint8)
        classified_array[canopy_array > 0] = 1
        return classified_array

    def process_indices(self, band_data, transform, polygon, reclassify_function, canopy=False):
        classified_image = reclassify_function(band_data)
        shapes_gen = shapes(classified_image, mask=None, transform=transform)
        geometries = list(shapes_gen)

        geojson_data = [
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {"canopy": bool(value)} if canopy else {"class_no": value}
            }
            for geom, value in geometries if value != 0
        ]

        geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
        geojson_data_df = gpd.GeoDataFrame.from_features({"type": "FeatureCollection", "features": geojson_data}, crs='epsg:4326')
        intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)

        return [
            {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": row.drop("geometry").to_dict()
            }
            for _, row in intersection_df.iterrows()
        ]


# Chlorophyll Forecast
class CARIFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []
        
        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception as e:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
            serializer = IndicesSerializer(data={'date': date})  # Adjust data as needed
            if serializer.is_valid():
                evalscript = """
                function setup() {
                    return {
                        input: ["B03", "B04", "B08", "B11", "SCL"], // Added B11 for better vegetation detection
                        output: {
                            id: "default",
                            bands: 1,
                            sampleType: "FLOAT32"
                        },
                        mosaicking: "ORBIT"
                    };
                }

                function preProcessScenes(collections) {
                    collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                        var orbitDateFrom = new Date(orbit.dateFrom);
                        return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000); // 3-month filter
                    });
                    return collections;
                }

                function getValue(values) {
                    values.sort(function (a, b) { return a - b; });
                    return getFirstQuartile(values);
                }

                function getFirstQuartile(sortedValues) {
                    var index = Math.floor(sortedValues.length / 4);
                    return sortedValues[index];
                }

                function validate(sample) {
                    var scl = sample.SCL;
                    // Exclude clouds, cloud shadows, and water
                    if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                        return false;
                    }
                    return true;
                }

                function evaluatePixel(samples, scenes) {
                    var validValuesB08 = [], validValuesB04 = [], validValuesB03 = [], validValuesB11 = [];
                    var a = 0;

                    for (var i = 0; i < samples.length; i++) {
                        var sample = samples[i];
                        if (sample.B08 > 0 && sample.B04 > 0 && sample.B03 > 0 && sample.B11 > 0) {
                            var isValid = validate(sample);
                            if (isValid) {
                                validValuesB08[a] = sample.B08;
                                validValuesB04[a] = sample.B04;
                                validValuesB03[a] = sample.B03;
                                validValuesB11[a] = sample.B11; // Using B11 for vegetation under trees
                                a++;
                            }
                        }
                    }

                    var CARI;
                    var canopyDetected = false;
                    if (a > 0) {
                        var GREEN = getValue(validValuesB03);
                        var RED = getValue(validValuesB04);
                        var NIR = getValue(validValuesB08);
                        var SWIR = getValue(validValuesB11); // Incorporating B11

                        // Calculate CARI considering vegetation under trees
                        var term1 = Math.pow((NIR - GREEN) / 150, 2);
                        var term2 = Math.pow((RED - GREEN), 2);
                        CARI = Math.sqrt(term1 + term2);

                        // Adjust CARI calculation for conditions under tree canopies
                        if (SWIR > 0.3) { // Threshold for identifying vegetation under canopies
                            CARI *= 1.1; // Example adjustment factor
                            canopyDetected = true; // Mark canopy detection
                        }
                    } else {
                        CARI = -9999; // No valid data
                    }

                    return [CARI, canopyDetected];
                }
                """

                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354.253],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue 

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_cari(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
                geojson_data = {"type": "FeatureCollection", "features": features}

                geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
                geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
                intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
                intersection_geojson = intersection_df.to_json()

                results.append(json.loads(intersection_geojson))

        predicted_results = self.predict_cari(results)
        return Response(predicted_results, status=status.HTTP_200_OK)

    def reclassify_cari(self, cari_array):
        classified_array = np.zeros_like(cari_array, dtype=np.uint8)
        classified_array[(cari_array <= 0)] = 1
        classified_array[(cari_array > 0) & (cari_array <= 0.1)] = 2
        classified_array[(cari_array > 0.1) & (cari_array <= 0.2)] = 3
        classified_array[(cari_array > 0.2) & (cari_array <= 0.4)] = 4
        classified_array[(cari_array > 0.4) & (cari_array <= 0.5)] = 5
        classified_array[(cari_array > 0.5) & (cari_array <= 0.6)] = 6
        classified_array[(cari_array > 0.6) & (cari_array <= 0.7)] = 7
        classified_array[(cari_array > 0.7)] = 8
        classified_array[(cari_array == -9999)] = 0  # Set cloudy pixels to 0
        return classified_array

    def predict_cari(self, results):
        predicted_features = []
        valid_coordinates = []
        valid_class_numbers = []
        valid_canopy_flags = []
        valid_results = []

        if not results or not isinstance(results, list):
            print("No results provided or results are not in the expected format.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features 
            }

        for feature in results:
            if 'features' not in feature:
                print("Feature missing 'features' key:", feature)
                continue

            for item in feature['features']:
                geometry = item.get('geometry', {})
                coords = geometry.get('coordinates', [])

                if isinstance(coords, list) and coords and isinstance(coords[0], list):
                    if len(coords[0]) > 0 and len(coords[0][0]) == 2:
                        valid_coord = coords[0][0]
                        valid_coordinates.append(valid_coord)
                        valid_results.append(item)

                        class_no = item.get('properties', {}).get('class_no')
                        canopy_detected = item.get('properties', {}).get('canopyDetected', False)
                        if class_no is not None:
                            valid_class_numbers.append(class_no)
                            valid_canopy_flags.append(canopy_detected)
                        else:
                            print("Class number missing in properties.")
                    else:
                        a = 3 #print(f"Invalid coordinate structure found: {coords}")
                else:
                    print(f"Invalid coordinate found: {coords}")

        if not valid_coordinates or not valid_class_numbers:
            print("No valid coordinates or class numbers found.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features
            }

        coordinates_array = np.array(valid_coordinates)
        class_numbers_array = np.array(valid_class_numbers)

        model = LinearRegression()
        model.fit(coordinates_array, class_numbers_array)
        predicted_class = model.predict(coordinates_array)

        for i, item in enumerate(valid_results):
            item['properties']['predicted_class'] = predicted_class[i]
            item['properties']['canopy_detected'] = valid_canopy_flags[i]

        return {
            "type": "FeatureCollection",
            "features": valid_results 
        }

# Chlorophyll Growth Forecast
class MCARIFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception as e:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
            serializer = IndicesSerializer(data={'date': date})
            if serializer.is_valid():
                cloud_coverage = int(request.data.get('cloud_coverage', 20))
                if cloud_coverage > 100:
                    return Response({'error': 'Cloud coverage value cannot be greater than 100.'}, status=status.HTTP_400_BAD_REQUEST)

                evalscript = """
                //VERSION=3
                function setup() {
                  return {
                    input: ["B02", "B03", "B04", "B08", "CLM"],
                    output: { bands: 1, sampleType: "FLOAT32" }
                  };
                }
                function evaluatePixel(sample) {
                  if (sample.CLM == 1) {
                    return [-9999];
                  }
                  var BLUE = sample.B02;
                  var GREEN = sample.B03;
                  var RED = sample.B04;
                  var NIR = sample.B08;
                  
                  // Calculate MCARI
                  return [(1.5 * (2.5 * (NIR - RED) - 1.3 * (NIR - GREEN)))];
                }
                """

                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354.253],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_mcari(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
                geojson_data = {"type": "FeatureCollection", "features": features}

                geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
                geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
                intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
                intersection_geojson = intersection_df.to_json()

                results.append(json.loads(intersection_geojson))

        predicted_results = self.predict_mcari(results)
        return Response(predicted_results, status=status.HTTP_200_OK)

    def reclassify_mcari(self, mcari_array):
        classified_array = np.zeros_like(mcari_array, dtype=np.uint8)
        classified_array[(mcari_array <= 0) & (mcari_array != -9999)] = 1
        classified_array[(mcari_array > 0) & (mcari_array <= 0.1)] = 2
        classified_array[(mcari_array > 0.1) & (mcari_array <= 0.2)] = 3
        classified_array[(mcari_array > 0.2) & (mcari_array <= 0.4)] = 4
        classified_array[(mcari_array > 0.4) & (mcari_array <= 0.5)] = 5
        classified_array[(mcari_array > 0.5) & (mcari_array <= 0.6)] = 6
        classified_array[(mcari_array > 0.6) & (mcari_array <= 0.7)] = 7
        classified_array[(mcari_array > 0.7) & (mcari_array <= 1)] = 8
        classified_array[(mcari_array == -9999)] = 0  # Cloudy pixels set to 0
        return classified_array

    def predict_mcari(self, results):
        predicted_features = []
        valid_coordinates = []
        valid_class_numbers = []
        valid_results = []

        if not results or not isinstance(results, list):
            print("No results provided or results are not in the expected format.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features 
            }

        for feature in results:
            if 'features' not in feature:
                print("Feature missing 'features' key:", feature)
                continue

            for item in feature['features']:
                geometry = item.get('geometry', {})
                coords = geometry.get('coordinates', [])

                if isinstance(coords, list) and coords and isinstance(coords[0], list):
                    if len(coords[0]) > 0 and len(coords[0][0]) == 2:
                        valid_coord = coords[0][0]
                        valid_coordinates.append(valid_coord)
                        valid_results.append(item)

                        class_no = item.get('properties', {}).get('class_no')
                        if class_no is not None:
                            valid_class_numbers.append(class_no)
                        else:
                            print("Class number missing in properties.")
                    else:
                        a = 3 #print(f"Invalid coordinate structure found: {coords}")
                else:
                    print(f"Invalid coordinate found: {coords}")

        if not valid_coordinates or not valid_class_numbers:
            print("No valid coordinates or class numbers found.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features
            }

        coordinates_array = np.array(valid_coordinates)
        class_numbers_array = np.array(valid_class_numbers)

        model = LinearRegression()
        model.fit(coordinates_array, class_numbers_array)

        max_sample_size = min(1000, len(coordinates_array))
        sampled_indices = random.sample(range(len(coordinates_array)), max_sample_size)

        sampled_coordinates = coordinates_array[sampled_indices]
        predicted_class_no = model.predict(sampled_coordinates).astype(int)

        predicted_features = [
            {
                "id": str(i),
                "type": "Feature",
                "properties": {
                    "class_no": class_no
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": valid_results[idx]['geometry']['coordinates']
                }
            }
            for i, (idx, class_no) in enumerate(zip(sampled_indices, predicted_class_no))
        ]

        predicted_geojson = {
            "type": "FeatureCollection",
            "features": predicted_features
        }

        return predicted_geojson

# Deforestration forecase
class DeforestationFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)

            evalscript = """
            function setup() {
                return { 
                    input: ["B04", "B08", "SCL"], 
                    output: { bands: 2, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function validate(sample) {
                var scl = sample.SCL;
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false;
                }
                return true;
            }

            function evaluatePixel(samples) {
                var validB08 = [];
                var validB04 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B04 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validB08[a] = sample.B08;
                            validB04[a] = sample.B04;
                            a++;
                        }
                    }
                }

                var ndvi = -9999;
                if (a > 0) {
                    var avgB08 = validB08.reduce((a, b) => a + b, 0) / validB08.length;
                    var avgB04 = validB04.reduce((a, b) => a + b, 0) / validB04.length;

                    ndvi = (avgB08 - avgB04) / (avgB08 + avgB04);
                }

                var class_no = 0;
                if (ndvi >= 0.5) {
                    class_no = 2;  // Forested
                } else if (ndvi >= 0.2) {
                    class_no = 1;  // Deforested
                }

                return [ndvi, class_no];
            }
            """

            try:
                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A, 
                        time_interval=(date, date)
                    )],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_deforestation(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [
                    {"type": "Feature", "geometry": geom, "properties": {"class_no": value}}
                    for geom, value in geometries if value != 0
                ]
                geojson_data = {"type": "FeatureCollection", "features": features}

                results.extend(self.process_geojson_data(geojson_data, polygon))

            except Exception as e:
                return Response({'error': f'Error in SentinelHub request: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"type": "FeatureCollection", "features": results}, status=status.HTTP_200_OK)

    def reclassify_deforestation(self, data_array):
        ndvi_array, class_array = data_array[..., 0], data_array[..., 1]
        classified_array = np.zeros_like(ndvi_array, dtype=np.uint8)

        classified_array[(class_array == 1)] = 1  # Deforested
        classified_array[(class_array == 2)] = 2  # Forested
        classified_array[(ndvi_array == -9999)] = 0  # Invalid pixels
        return classified_array

    def process_geojson_data(self, geojson_data, polygon):
        try:
            # Ensure the polygon is a valid geometry and create the GeoDataFrame
            geojson_polygon_df = gpd.GeoDataFrame({'geometry': [polygon]}, crs='epsg:4326')
            
            # Check if the polygon GeoDataFrame is valid
            if geojson_polygon_df.is_empty.any():
                return []

            # Ensure the GeoJSON data contains the 'features' field and is valid
            if 'features' not in geojson_data or not geojson_data['features']:
                return []

            # Create a GeoDataFrame from the GeoJSON features
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data['features'], crs='epsg:4326')

            # Ensure the GeoDataFrame has a valid 'geometry' column
            if geojson_data_df.empty or 'geometry' not in geojson_data_df.columns:
                return []

            # Perform the intersection with the polygon
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df, how='intersection')

            # Generate features from the intersection result
            features = []
            for _, row in intersection_df.iterrows():
                feature = {
                    "type": "Feature",
                    "geometry": mapping(row.geometry),  # Convert geometry to GeoJSON format
                    "properties": {key: value for key, value in row.items() if key != 'geometry'}
                }
                features.append(feature)

            return features

        except Exception as e:
            # Handle any exceptions
            raise ValueError(f"Error processing GeoJSON data: {e}")


# Disease Biomass Forecast
class TVIFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)

            evalscript = """
            function setup() {
                return { 
                    input: ["B05", "B08", "SCL"], 
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function validate(sample) {
                var scl = sample.SCL;
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false;
                }
                return true;
            }

            function evaluatePixel(samples) {
                var validB08 = [];
                var validB05 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B05 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validB08[a] = sample.B08;
                            validB05[a] = sample.B05;
                            a++;
                        }
                    }
                }

                var tvif;
                if (a > 0) {
                    var avgB08 = validB08.reduce((a, b) => a + b, 0) / validB08.length;
                    var avgB05 = validB05.reduce((a, b) => a + b, 0) / validB05.length;

                    tvif = (avgB08 - avgB05) / (avgB08 + avgB05);
                } else {
                    tvif = -9999;
                }

                return [tvif];
            }
            """

            try:
                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=(date, date)
                    )],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_tvif(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [
                    {"type": "Feature", "geometry": geom, "properties": {"class_no": value}}
                    for geom, value in geometries if value != 0
                ]
                geojson_data = {"type": "FeatureCollection", "features": features}

                results.extend(self.process_geojson_data(geojson_data, polygon))

            except Exception as e:
                return Response({'error': f'Error in SentinelHub request: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"type": "FeatureCollection", "features": results}, status=status.HTTP_200_OK)

    def reclassify_tvif(self, tvif_array):
        classified_array = np.zeros_like(tvif_array, dtype=np.uint8)
        classified_array[(tvif_array <= 0) & (tvif_array != -9999)] = 1
        classified_array[(tvif_array > 0) & (tvif_array <= 0.1)] = 2
        classified_array[(tvif_array > 0.1) & (tvif_array <= 0.2)] = 3
        classified_array[(tvif_array > 0.2) & (tvif_array <= 0.4)] = 4
        classified_array[(tvif_array > 0.4) & (tvif_array <= 0.5)] = 5
        classified_array[(tvif_array > 0.5) & (tvif_array <= 0.6)] = 6
        classified_array[(tvif_array > 0.6) & (tvif_array <= 0.7)] = 7
        classified_array[(tvif_array > 0.7)] = 8
        classified_array[(tvif_array == -9999)] = 0
        return classified_array

    def process_geojson_data(self, geojson_data, polygon):
        geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
        geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
        intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)

        features = []
        for _, row in intersection_df.iterrows():
            features.append({
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": row.drop("geometry").to_dict()
            })

        return features


# Disease Detection Forecase(EVI)
class EVIFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)

            evalscript = """
            function setup() {
                return { 
                    input: ["B03", "B08", "SCL"], 
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function validate(sample) {
                var scl = sample.SCL;
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false;
                }
                return true;
            }

            function evaluatePixel(samples) {
                var validB08 = [];
                var validB03 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B03 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validB08[a] = sample.B08;
                            validB03[a] = sample.B03;
                            a++;
                        }
                    }
                }

                var evif;
                if (a > 0) {
                    var avgB08 = validB08.reduce((a, b) => a + b, 0) / validB08.length;
                    var avgB03 = validB03.reduce((a, b) => a + b, 0) / validB03.length;

                    evif = (avgB08 - avgB03) / (avgB08 + avgB03);
                } else {
                    evif = -9999;
                }

                return [evif];
            }
            """

            try:
                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=(date, date)
                    )],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_evif(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [
                    {"type": "Feature", "geometry": geom, "properties": {"class_no": value}}
                    for geom, value in geometries if value != 0
                ]
                geojson_data = {"type": "FeatureCollection", "features": features}

                results.extend(self.process_geojson_data(geojson_data, polygon))

            except Exception as e:
                return Response({'error': f'Error in SentinelHub request: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"type": "FeatureCollection", "features": results}, status=status.HTTP_200_OK)

    def reclassify_evif(self, evif_array):
        classified_array = np.zeros_like(evif_array, dtype=np.uint8)
        classified_array[(evif_array <= 0) & (evif_array != -9999)] = 1
        classified_array[(evif_array > 0) & (evif_array <= 0.1)] = 2
        classified_array[(evif_array > 0.1) & (evif_array <= 0.2)] = 3
        classified_array[(evif_array > 0.2) & (evif_array <= 0.4)] = 4
        classified_array[(evif_array > 0.4) & (evif_array <= 0.5)] = 5
        classified_array[(evif_array > 0.5) & (evif_array <= 0.6)] = 6
        classified_array[(evif_array > 0.6) & (evif_array <= 0.7)] = 7
        classified_array[(evif_array > 0.7)] = 8
        classified_array[(evif_array == -9999)] = 0
        return classified_array

    def process_geojson_data(self, geojson_data, polygon):
        geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
        geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
        intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)

        features = []
        for _, row in intersection_df.iterrows():
            features.append({
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": row.drop("geometry").to_dict()
            })

        return features
