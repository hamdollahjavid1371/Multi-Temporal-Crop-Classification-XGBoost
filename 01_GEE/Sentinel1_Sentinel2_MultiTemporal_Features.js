// =====================================================
// Multi-Temporal Sentinel-1 and Sentinel-2
// Crop Classification Feature Extraction
// =====================================================

// =====================================================
// 1. Study Area
// =====================================================

// Replace this asset with your own study-area boundary.
var studyArea = ee.FeatureCollection(
  'YOUR_STUDY_AREA_ASSET'
);

var roi = studyArea.geometry();

Map.centerObject(studyArea, 10);

Map.addLayer(
  studyArea,
  {color: 'red'},
  'Study Area'
);


// =====================================================
// 2. Sentinel-2 Collection
// =====================================================

var s2 = ee.ImageCollection(
  'COPERNICUS/S2_SR_HARMONIZED'
)
.filterBounds(roi)
.filterDate(
  '2025-10-01',
  '2026-06-01'
)
.filter(
  ee.Filter.lt(
    'CLOUDY_PIXEL_PERCENTAGE',
    40
  )
);


// =====================================================
// 3. Sentinel-2 Cloud Mask
// =====================================================

function maskS2(image) {

  var qa = image.select('QA60');

  var cloud = qa
    .bitwiseAnd(1 << 10)
    .eq(0)
    .and(
      qa
        .bitwiseAnd(1 << 11)
        .eq(0)
    );

  return image
    .updateMask(cloud)
    .divide(10000)
    .copyProperties(
      image,
      ['system:time_start']
    );
}

var s2clean = s2.map(maskS2);


// =====================================================
// 4. NDVI and NDRE
// =====================================================

function addIndices(image) {

  var ndvi = image
    .normalizedDifference([
      'B8',
      'B4'
    ])
    .rename('NDVI');

  var ndre = image
    .normalizedDifference([
      'B8',
      'B5'
    ])
    .rename('NDRE');

  return image
    .addBands(ndvi)
    .addBands(ndre);
}

var s2index = s2clean.map(addIndices);


// =====================================================
// 5. Sentinel-1 Collection
// =====================================================

var s1 = ee.ImageCollection(
  'COPERNICUS/S1_GRD'
)
.filterBounds(roi)
.filterDate(
  '2025-10-01',
  '2026-06-01'
)
.filter(
  ee.Filter.eq(
    'instrumentMode',
    'IW'
  )
)
.filter(
  ee.Filter.listContains(
    'transmitterReceiverPolarisation',
    'VV'
  )
)
.filter(
  ee.Filter.listContains(
    'transmitterReceiverPolarisation',
    'VH'
  )
);

print(
  'Sentinel-2 images',
  s2index.size()
);

print(
  'Sentinel-1 images',
  s1.size()
);


// =====================================================
// 6. Monthly Feature Stack
// =====================================================

function monthlyStack(
  month,
  start,
  end
) {

  var optical = s2index
    .filterDate(start, end)
    .median()
    .clip(roi);

  var radar = s1
    .filterDate(start, end)
    .median()
    .clip(roi);

  var ndvi = optical
    .select('NDVI')
    .rename('NDVI_' + month);

  var ndre = optical
    .select('NDRE')
    .rename('NDRE_' + month);

  var vv = radar
    .select('VV')
    .rename('VV_' + month);

  var vh = radar
    .select('VH')
    .rename('VH_' + month);

  var vvvh = radar
    .select('VV')
    .subtract(
      radar.select('VH')
    )
    .rename('VVVH_' + month);

  return ndvi
    .addBands(ndre)
    .addBands(vv)
    .addBands(vh)
    .addBands(vvvh);
}


// =====================================================
// 7. Create 40-Band Multi-Temporal Stack
// =====================================================

var stack = monthlyStack(
  'Mehr',
  '2025-10-01',
  '2025-11-01'
)

.addBands(
  monthlyStack(
    'Aban',
    '2025-11-01',
    '2025-12-01'
  )
)

.addBands(
  monthlyStack(
    'Azar',
    '2025-12-01',
    '2026-01-01'
  )
)

.addBands(
  monthlyStack(
    'Dey',
    '2026-01-01',
    '2026-02-01'
  )
)

.addBands(
  monthlyStack(
    'Bahman',
    '2026-02-01',
    '2026-03-01'
  )
)

.addBands(
  monthlyStack(
    'Esfand',
    '2026-03-01',
    '2026-04-01'
  )
)

.addBands(
  monthlyStack(
    'Farvardin',
    '2026-04-01',
    '2026-05-01'
  )
)

.addBands(
  monthlyStack(
    'Ordibehesht',
    '2026-05-01',
    '2026-06-01'
  )
);

stack = stack.toFloat();


// =====================================================
// 8. Check Stack
// =====================================================

print(
  'Stack bands:',
  stack.bandNames()
);

print(
  'Number of bands:',
  stack.bandNames().size()
);


// =====================================================
// 9. Display Example
// =====================================================

Map.addLayer(
  stack,
  {
    bands: [
      'NDVI_Mehr',
      'NDVI_Dey',
      'NDVI_Ordibehesht'
    ],
    min: 0,
    max: 1
  },
  'NDVI Multi-Temporal Stack'
);


// =====================================================
// 10. Export 40-Band GeoTIFF
// =====================================================

Export.image.toDrive({

  image: stack,

  description:
    'MultiTemporal_S1_S2_40Features',

  folder:
    'GEE_Stack',

  fileNamePrefix:
    'MultiTemporal_S1_S2_40Features',

  region:
    roi,

  scale:
    10,

  crs:
    'EPSG:32639',

  maxPixels:
    1e13,

  fileFormat:
    'GeoTIFF'
});
