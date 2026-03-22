// ===============================
// 1. LOAD NUREMBERG BOUNDARY
// ===============================

var nuremberg = ee.FeatureCollection(
  "projects/nuremberg-landcover-ml/assets/nuremberg_boundary"
);

Map.centerObject(nuremberg, 11);
Map.addLayer(nuremberg, {color: 'red'}, "Nuremberg Boundary");


// ===============================
// 2. LOAD SENTINEL-2 IMAGERY
// ===============================

var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(nuremberg)
  .filterDate('2021-01-01', '2021-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .select(['B2','B3','B4','B8','B11'])
  .median()
  .clip(nuremberg);


// Visualization
var vis = {
  bands: ['B4','B3','B2'],
  min: 0,
  max: 3000
};

Map.addLayer(s2, vis, "Sentinel-2 2018");


// ===============================
// 3. CALCULATE INDICES
// ===============================

// NDVI
var ndvi = s2.normalizedDifference(['B8','B4']).rename('NDVI');

// NDBI
var ndbi = s2.normalizedDifference(['B11','B8']).rename('NDBI');

// NDWI
var ndwi = s2.normalizedDifference(['B3','B8']).rename('NDWI');


// Visualization
Map.addLayer(ndvi, {min:-1,max:1,palette:['blue','white','green']}, "NDVI");
Map.addLayer(ndbi, {min:-1,max:1,palette:['green','white','red']}, "NDBI");
Map.addLayer(ndwi, {min:-1,max:1,palette:['brown','white','blue']}, "NDWI");


// ===============================
// 4. LOAD LAND COVER LABELS
// ===============================

var esa = ee.Image("ESA/WorldCover/v200/2021")
            .select('Map')
            .clip(nuremberg);

// Visualization
var corineVis = {
  min: 1,
  max: 44,
  palette: [
    'e6004d','ff0000','ff4d4d','ff9999',
    'ffffa8','ffff00','e6e600','999900',
    'ccffcc','a6e64d','4dff00','00ff00',
    '00a600','4dff4d','b3ffb3','d1ffd1',
    'a6e6ff','4da6ff','0064ff','0040ff'
  ]
};

Map.addLayer(esa, {}, "ESA WorldCover 2021");


// ===============================
// 5. CREATE FEATURE STACK
// ===============================

var features = s2
  .addBands(ndvi)
  .addBands(ndbi)
  .addBands(ndwi);


// Combine features + labels
var dataset = features.addBands(esa.rename('label'));


// ===============================
// 6. CREATE 200m GRID
// ===============================

var grid = nuremberg.geometry().coveringGrid({
  proj: ee.Projection('EPSG:3857').atScale(200),
  scale: 200
});

Map.addLayer(grid, {color:'yellow'}, "Grid 200m");


// ===============================
// 7. COMPUTE GRID STATISTICS
// ===============================

// mean reducer for features
var featureReducer = ee.Reducer.mean();

// mode reducer for labels
var labelReducer = ee.Reducer.mode();

// combine reducers
var combinedReducer = featureReducer.combine({
  reducer2: labelReducer,
  sharedInputs: true
});


// apply reduceRegions
var gridStats = dataset.reduceRegions({
  collection: grid,
  reducer: combinedReducer,
  scale: 10
});


// ===============================
// 8. CLEAN COLUMNS
// ===============================

var cleanData = gridStats.select([
  'B2_mean',
  'B3_mean',
  'B4_mean',
  'B8_mean',
  'B11_mean',
  'NDVI_mean',
  'NDBI_mean',
  'NDWI_mean',
  'label_mode'
],[ 
  'B2',
  'B3',
  'B4',
  'B8',
  'B11',
  'NDVI',
  'NDBI',
  'NDWI',
  'label'
]);


// Preview
print("First 10 grid cells", cleanData.limit(10));


// ===============================
// 9. EXPORT DATASET
// ===============================

Export.table.toDrive({
  collection: cleanData,
  description: 'nuremberg_grid_dataset_2021_ESA_200m',
  fileFormat: 'CSV'
});