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
  .filterDate('2018-01-01', '2018-12-31')   // change year when needed
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

Map.addLayer(s2, vis, "Sentinel-2");


// ===============================
// 3. CALCULATE INDICES
// ===============================

// NDVI
var ndvi = s2.normalizedDifference(['B8','B4']).rename('NDVI');

// NDBI
var ndbi = s2.normalizedDifference(['B11','B8']).rename('NDBI');

// NDWI
var ndwi = s2.normalizedDifference(['B3','B8']).rename('NDWI');


// Visualizations
Map.addLayer(ndvi, {min:-1,max:1,palette:['blue','white','green']}, "NDVI");
Map.addLayer(ndbi, {min:-1,max:1,palette:['green','white','red']}, "NDBI");
Map.addLayer(ndwi, {min:-1,max:1,palette:['brown','white','blue']}, "NDWI");


// ===============================
// 4. LOAD LAND-COVER LABELS
// ===============================

var worldcover = ee.Image("ESA/WorldCover/v100/2020")
                    .select('Map')
                    .clip(nuremberg);


// Visualization
var worldcoverVis = {
  min: 10,
  max: 100,
  palette: [
    '006400','ffbb22','ffff4c','f096ff',
    'fa0000','b4b4b4','f0f0f0','0064c8',
    '0096a0','00cf75','fae6a0'
  ]
};

Map.addLayer(worldcover, worldcoverVis, "ESA WorldCover");


// ===============================
// 5. CREATE FEATURE STACK
// ===============================

var features = s2
  .addBands(ndvi)
  .addBands(ndbi)
  .addBands(ndwi);


// Combine features + labels
var dataset = features.addBands(worldcover.rename('label'));


// ===============================
// 6. CREATE 300m GRID
// ===============================

var grid = nuremberg.geometry().coveringGrid({
  proj: ee.Projection('EPSG:3857').atScale(300),
  scale: 300
});

Map.addLayer(grid, {color:'yellow'}, "Grid");


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


// preview
print("First 10 grid cells", cleanData.limit(10));


// ===============================
// 9. EXPORT DATASET
// ===============================

Export.table.toDrive({
  collection: cleanData,
  description: 'nuremberg_grid_dataset_2018',
  fileFormat: 'CSV'
});