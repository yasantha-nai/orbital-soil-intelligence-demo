"""
Sentinel-2 optical and Sentinel-1 SAR data collectors.

Sentinel-2: Spectral indices + ESA SNAP S2 Biophysical Processor v2.1
Sentinel-1: VH, VV, RVI radar backscatter

Enhanced with 3-layer cloud masking (QA60 + SCL + MSK_CLDPRB).
"""

import math
import datetime as _dt
from datetime import date as _date

import ee
import pandas as pd

from config import MAX_CLOUD_COVER, S2_SELECT_BANDS, S2_SCALE


# ── ESA SNAP S2 Biophysical NN weights (ATBD v2.1) ──────────────────────────
# Architecture: 11 inputs → 5 hidden (tanh) → 1 output (linear) → denormalise
# Input order: B3, B4, B5, B6, B7, B8A, B11, B12, cos(vz), cos(sz), cos(raz)

_SNAP_NORM_MIN = [
    0.0, 0.0, 0.0, 0.00663797254225, 0.0139727270189, 0.0266901380821,
    0.0163880741923, 0.0, 0.918595400582, 0.342022871159, -0.999999982118,
]
_SNAP_NORM_MAX = [
    0.253061520472, 0.290393577911, 0.305398915249, 0.608900395798,
    0.753827384323, 0.782011770669, 0.493761397883, 0.49302598446,
    0.999999999991, 0.936206429175, 0.99999999891,
]
_SNAP_DEFDOM_MIN = [0.0, 0.0, 0.0, 0.0, 0.00397272701894, 0.0166901380821, 0.00638807419226, 0.0]
_SNAP_DEFDOM_MAX = [0.263061520472, 0.300393577911, 0.315398915249, 0.618900395798,
                    0.763827384323, 0.792011770669, 0.503761397883, 0.50302598446]

_S2_BIOPHYSICAL_NN = {
    "LAI": {
        "W1": [
            [-0.0234068789665, 0.921655164636, 0.13557654408, -1.9383314724, -3.34249581612,
              0.90227764801, 0.205363538259, -0.0406078447217, -0.0831964097271, 0.260029270774, 0.284761567219],
            [-0.132555480857, -0.139574837334, -1.0146060169, -1.33089003865, 0.0317306245033,
             -1.43358354132, -0.959637898575, 1.13311570655, 0.216603876542, 0.410652303763, 0.0647601555435],
            [0.0860159777249, 0.616648776881, 0.678003876447, 0.141102398645, -0.0966822068835,
             -1.12883263886, 0.302189102741, 0.4344949373, -0.0219036994906, -0.228492476802, -0.0394605375898],
            [-0.10936659367, -0.0710462629727, 0.0645824114783, 2.90632523682, -0.673873108979,
             -3.83805186828, 1.69597934453, 0.0469502960817, -0.0497096526884, 0.021829545431, 0.0574838271041],
            [-0.08993941616, 0.175395483106, -0.0818473291726, 2.21989536749, 1.71387397514,
              0.7130691861, 0.138970813499, -0.060771761518, 0.124263341255, 0.210086140404, -0.1838781387],
        ],
        "b1": [4.96238030555, 1.41600844398, 1.07589704721, 1.53398826466, 3.02411593076],
        "W2": [[-1.50013548973, -0.0962832691215, -0.194935930577, -0.352305895756, 0.0751074158475]],
        "b2": [1.09696310708],
        "denorm_min": 0.000319182538301, "denorm_max": 14.4675094548,
        "tolerance": -0.2, "out_min": 0.0, "out_max": 8.0, "output_name": "LAI",
    },
    "LAI_Cab": {
        "W1": [
            [0.400396555257, 0.607936279259, 0.13746865078, -2.95586657346, -3.18674668773,
              2.20680075125, -0.31378433614, 0.256063547511, -0.0716132198051, 0.51011350421, 0.142813982139],
            [-0.250781102415, 0.43908630292, -1.16059093752, -1.86193525027, 0.981359868452,
              1.63423083425, -0.872527934646, 0.448240475035, 0.0370780835012, 0.0300441896704, 0.0059566866194],
            [0.552080132569, -0.502919673167, 6.10504192497, -1.29438611914, -1.05995638835,
             -1.39409290242, 0.324752732711, -1.75887182283, -0.0366636798603, -0.183105291401, -0.0381453121174],
            [0.211591184882, -0.248788896074, 0.887151598039, 1.14367589557, -0.753968830338,
             -1.18545695308, 0.541897860472, -0.252685834608, -0.0234149010781, -0.0460225035496, -0.00657028408066],
            [0.254790234231, -0.724968611431, 0.731872806027, 2.30345382102, -0.849907966922,
             -6.42531550054, 2.23884455846, -0.199937574298, 0.0973033317146, 0.334528254938, 0.113075306592],
        ],
        "b1": [4.24229967016, -0.259569088226, 3.13039262734, 0.774423577182, 2.58427664853],
        "W2": [[-0.352760040599, -0.603407399151, 0.135099379384, -1.73567312385, -0.147546813318]],
        "b2": [0.463426463934],
        "denorm_min": 0.00742669295987, "denorm_max": 873.90822211,
        "tolerance": -15.0, "out_min": 0.0, "out_max": 600.0, "output_name": "CCC",
    },
    "LAI_Cw": {
        "W1": [
            [0.146378710426, 1.18979928187, -0.906235139963, -0.808337508767, -0.97333491783,
             -1.42591277646, -0.00561253629588, -0.634520356267, -0.117226059989, -0.0602700912102, 0.229407587132],
            [0.283319173374, 0.149342023041, 1.08480588387, -0.138658791035, -0.455759407329,
              0.420571438078, -1.7372949037, -0.704286287226, 0.0190953782358, -0.0393971316513, -0.00750241581744],
            [-0.197487427943, -0.105460325978, 0.158347670681, 2.14912426654, -0.970716842916,
             -4.92725317909, 1.42034301781, 1.45316917226, 0.0227257053609, 0.269298650421, 0.0849047657715],
            [0.141405799763, 0.33386260328, 0.356218929123, -0.545942267639, 0.0891043076856,
              0.919298362929, -1.8520892625, -0.427539590779, 0.00791385646467, 0.0148333201478, -0.00153786769736],
            [-0.186781083395, -0.549163704901, -0.181287638772, 0.96864043656, -0.470442559117,
             -1.24859725244, 2.67014942338, 0.49009062438, -0.00144931939526, 0.00314829369692, 0.0206517883893],
        ],
        "b1": [-2.1064083686, -1.69022094794, 3.10117655255, -1.31231626496, 1.01131930348],
        "W2": [[-0.0775555890347, -0.86411786119, -0.199212415374, 1.98730461219, 0.458926743489]],
        "b2": [-0.197591709977],
        "denorm_min": 3.85066859366e-06, "denorm_max": 0.522417054645,
        "tolerance": -0.015, "out_min": 0.0, "out_max": 0.55, "output_name": "CWC",
    },
    "FAPAR": {
        "W1": [
            [0.268714454733, -0.20547310803, 0.281765694196, 1.33744341226, 0.390319212938,
             -3.6127143422, 0.222530960987, 0.821790549667, -0.0936645673107, 0.0192901461474, 0.0373644463772],
            [-0.2489980546, -0.571461305473, -0.369957603467, 0.246031694651, 0.332536215253,
              0.438269896209, 0.81900055189, -0.934931499059, 0.0827162476519, -0.286978634108, -0.0358909683517],
            [-0.164063575316, -0.126303285738, -0.253670784367, -0.321162835049, 0.0670822879736,
              2.02983228866, -0.0231412288277, -0.553176625658, 0.0592854518978, -0.0343344545414, -0.031776704097],
            [0.130240753004, 0.236781035723, 0.131811664093, -0.250181799268, -0.0113641499533,
             -1.85757321463, -0.146860751014, 0.528008831372, -0.0462307690983, -0.0345096083922, 0.031884395036],
            [-0.0299299461669, 0.795804414041, 0.348025317625, 0.943567007519, -0.276341670432,
             -2.94659418014, 0.289483073507, 1.04400695044, -0.000413031960419, 0.40333111484, 0.0684271305267],
        ],
        "b1": [-0.88706836404, 0.320126471197, 0.6105237025, -0.379156190834, 1.35302339669],
        "W2": [[2.12603881106, -0.632044932795, 5.59899578721, 1.77044414058, -0.267879583605]],
        "b2": [-0.336431283973],
        "denorm_min": 0.000153013463222, "denorm_max": 0.97713509698,
        "tolerance": -0.1, "out_min": 0.0, "out_max": 0.94, "output_name": "FAPAR",
    },
    "FCOVER": {
        "W1": [
            [-0.156854264841, 0.124234528462, 0.235625516229, -1.8323910258, -0.217188969888,
              5.06933958064, -0.887578008155, -1.0808468167, -0.0323167041864, -0.224476137359, -0.195523962947],
            [-0.220824927842, 1.28595395487, 0.703139486363, -1.34481216665, -1.96881267559,
             -1.45444681639, 1.02737560043, -0.12494641532, 0.0802762437265, -0.198705918577, 0.108527100527],
            [-0.409688743281, 1.08858884766, 0.36284522554, 0.0369390509705, -0.348012590003,
             -2.0035261881, 0.0410357601757, 1.22373853174, -0.0124082778287, -0.282223364524, 0.0994993117557],
            [-0.188970957866, -0.0358621840833, 0.00551248528107, 1.35391570802, -0.739689896116,
             -2.21719530107, 0.313216124198, 1.5020168915, 1.21530490195, -0.421938358618, 1.48852484547],
            [2.49293993709, -4.40511331388, -1.91062012624, -0.703174115575, -0.215104721138,
             -0.972151494818, -0.930752241278, 1.2143441876, -0.521665460192, -0.445755955598, 0.344111873777],
        ],
        "b1": [-1.45261652206, -1.70417477557, 1.02168965849, -0.498002810205, -3.88922154789],
        "W2": [[0.23080586765, -0.333655484884, -0.499418292325, 0.0472484396749, -0.0798516540739]],
        "b2": [-0.0967998147811],
        "denorm_min": 0.000181230723879, "denorm_max": 0.999638214715,
        "tolerance": -0.1, "out_min": 0.0, "out_max": 1.0, "output_name": "FCOVER",
    },
}


# ── Cloud masking ─────────────────────────────────────────────────────────────

def _mask_s2_clouds(image: ee.Image) -> ee.Image:
    """3-layer cloud masking: QA60 + SCL + MSK_CLDPRB."""
    qa = image.select("QA60")
    qa_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))

    scl = image.select("SCL")
    bad_scl = [0, 1, 3, 8, 9, 10, 11]
    scl_mask = scl.neq(bad_scl[0])
    for cls in bad_scl[1:]:
        scl_mask = scl_mask.And(scl.neq(cls))

    cld_prob = image.select("MSK_CLDPRB")
    prob_mask = cld_prob.lt(40)

    return image.updateMask(qa_mask.And(scl_mask).And(prob_mask))


# ── Vegetation indices ────────────────────────────────────────────────────────

def _add_vegetation_indices(image: ee.Image) -> ee.Image:
    """Compute 9 spectral indices from S2 bands."""
    nir = image.select("B8")
    nir2 = image.select("B8A")
    red = image.select("B4")
    blue = image.select("B2")
    re1 = image.select("B5")
    re2 = image.select("B6")
    swir1 = image.select("B11")
    swir2 = image.select("B12")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    ndre = nir.subtract(re1).divide(nir.add(re1)).rename("NDRE")
    evi = image.expression(
        "2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)",
        {"NIR": nir, "RED": red, "BLUE": blue},
    ).rename("EVI")
    reip = image.expression(
        "700.0 + 40.0 * ((RED + NIR2) / 2.0 - RE1) / (RE2 - RE1)",
        {"RED": red, "NIR2": nir2, "RE1": re1, "RE2": re2},
    ).rename("REIP")
    ndwi = nir2.subtract(swir1).divide(nir2.add(swir1)).rename("NDWI")
    nbr = nir2.subtract(swir2).divide(nir2.add(swir2)).rename("NBR")
    ndii = nir2.subtract(swir1).divide(nir2.add(swir1)).rename("NDII")
    bsi = image.expression(
        "((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))",
        {"SWIR1": swir1, "RED": red, "NIR": nir, "BLUE": blue},
    ).rename("BSI")
    msr = swir1.divide(swir2.add(1e-6)).rename("MSR")

    return image.addBands([ndvi, ndre, evi, reip, ndwi, nbr, ndii, bsi, msr])


# ── Biophysical NN forward pass ───────────────────────────────────────────────

def _run_biophysical_nn(x_norm: ee.Image, nn: dict) -> ee.Image:
    W1 = nn["W1"]
    b1 = nn["b1"]
    W2 = nn["W2"][0]
    b2 = nn["b2"][0]

    hidden = []
    for i in range(5):
        weighted = None
        for j in range(11):
            term = x_norm.select(j).multiply(W1[i][j])
            weighted = term if weighted is None else weighted.add(term)
        hidden.append(weighted.add(b1[i]).tanh())

    output = None
    for j in range(5):
        term = hidden[j].multiply(W2[j])
        output = term if output is None else output.add(term)
    return output.add(b2)


def _add_biophysical_variables(image: ee.Image) -> ee.Image:
    """Compute LAI, CCC, CWC, FAPAR, FCOVER via ESA SNAP NNs."""
    pi = math.pi
    dn_scale = 10000.0

    b3 = image.select("B3").divide(dn_scale)
    b4 = image.select("B4").divide(dn_scale)
    b5 = image.select("B5").divide(dn_scale)
    b6 = image.select("B6").divide(dn_scale)
    b7 = image.select("B7").divide(dn_scale)
    b8a = image.select("B8A").divide(dn_scale)
    b11 = image.select("B11").divide(dn_scale)
    b12 = image.select("B12").divide(dn_scale)

    vz_deg = ee.Number(image.get("MEAN_INCIDENCE_ZENITH_ANGLE_B8A"))
    sz_deg = ee.Number(image.get("MEAN_SOLAR_ZENITH_ANGLE"))
    sa_deg = ee.Number(image.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
    va_deg = ee.Number(image.get("MEAN_INCIDENCE_AZIMUTH_ANGLE_B8A"))

    cos_vz = ee.Image.constant(vz_deg.multiply(pi / 180).cos())
    cos_sz = ee.Image.constant(sz_deg.multiply(pi / 180).cos())
    cos_raz = ee.Image.constant(sa_deg.subtract(va_deg).multiply(pi / 180).cos())

    spec_bands = [b3, b4, b5, b6, b7, b8a, b11, b12]

    defdom_mask = ee.Image.constant(1)
    for j, band in enumerate(spec_bands):
        defdom_mask = defdom_mask.And(
            band.gte(_SNAP_DEFDOM_MIN[j]).And(band.lte(_SNAP_DEFDOM_MAX[j]))
        )

    raw_inputs = [b3, b4, b5, b6, b7, b8a, b11, b12, cos_vz, cos_sz, cos_raz]
    norm_inputs = [
        raw_inputs[i].subtract(_SNAP_NORM_MIN[i])
        .divide(_SNAP_NORM_MAX[i] - _SNAP_NORM_MIN[i])
        .multiply(2.0).subtract(1.0)
        for i in range(11)
    ]
    x_norm = ee.Image.cat(norm_inputs)

    result_bands = []
    for nn in _S2_BIOPHYSICAL_NN.values():
        raw_out = _run_biophysical_nn(x_norm, nn)
        denormed = raw_out.add(1.0).multiply(0.5 * (nn["denorm_max"] - nn["denorm_min"]))
        tol, out_min, out_max = nn["tolerance"], nn["out_min"], nn["out_max"]
        too_low = denormed.lt(out_min + tol)
        too_high = denormed.gt(out_max - tol)
        denormed = denormed.max(out_min).min(out_max)
        out_valid = too_low.Not().And(too_high.Not())
        denormed = denormed.updateMask(defdom_mask).updateMask(out_valid)
        result_bands.append(denormed.rename(nn["output_name"]))

    return image.addBands(result_bands)


# ── Batch helper ──────────────────────────────────────────────────────────────

def _month_batches(start: str, end: str, months: int = 6):
    cur = _dt.datetime.strptime(start, "%Y-%m-%d").date()
    fin = _dt.datetime.strptime(end, "%Y-%m-%d").date()
    while cur < fin:
        m = cur.month - 1 + months
        nxt = _date(cur.year + m // 12, m % 12 + 1, 1)
        nxt = min(nxt, fin)
        yield cur.isoformat(), nxt.isoformat()
        cur = nxt


# ── Public API ────────────────────────────────────────────────────────────────

def collect_sentinel2(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    scale: int = S2_SCALE,
) -> pd.DataFrame:
    """
    Collect Sentinel-2 time series with spectral indices and biophysical variables.

    Returns one row per cloud-free scene with mean + std for each band.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER))
        .map(_mask_s2_clouds)
        .select(S2_SELECT_BANDS)
        .map(_add_vegetation_indices)
        .map(_add_biophysical_variables)
    )

    spectral_bands = ["NDVI", "NDRE", "EVI", "REIP", "NDWI", "NBR", "NDII", "BSI", "MSR"]
    bio_bands = ["LAI", "CCC", "CWC", "FAPAR", "FCOVER"]
    all_bands = spectral_bands + bio_bands

    combined_reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True)
    )

    def extract_stats(image):
        stats = image.select(all_bands).reduceRegion(
            reducer=combined_reducer, geometry=geometry, scale=scale, maxPixels=1e9,
        )
        date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        props = {"date": date, "pixel_count": stats.get("NDVI_count")}
        for b in all_bands:
            props[b] = stats.get(f"{b}_mean")
            props[f"{b}_std"] = stats.get(f"{b}_stdDev")
        return ee.Feature(None, props)

    features = []
    for batch_start, batch_end in _month_batches(start_date, end_date):
        batch_col = collection.filterDate(batch_start, batch_end)
        try:
            batch_features = batch_col.map(extract_stats).getInfo()["features"]
            features.extend(batch_features)
        except Exception as exc:
            print(f"[collect_sentinel2] batch {batch_start}–{batch_end} failed: {exc}")

    rows = []
    for f in features:
        p = f["properties"]
        if p.get("NDVI") is not None:
            row = {"date": p["date"], "pixel_count": int(p.get("pixel_count") or 0)}
            for b in all_bands:
                row[b] = round(p.get(b) or 0, 4) if p.get(b) is not None else None
                row[f"{b}_std"] = round(p.get(f"{b}_std") or 0, 4) if p.get(f"{b}_std") is not None else None
            rows.append(row)

    if not rows:
        cols = ["date", "pixel_count"] + [c for b in all_bands for c in [b, f"{b}_std"]] + ["NDVI_growth_rate"]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["NDVI_growth_rate"] = df["NDVI"].diff() / df["date"].diff().dt.days
    return df


def collect_sentinel1(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    scale: int = 10,
) -> pd.DataFrame:
    """Collect Sentinel-1 SAR (VH, VV, RVI) time series."""
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .select(["VV", "VH"])
    )

    def add_sar_indices(img):
        vh = img.select("VH")
        vv = img.select("VV")
        rvi = vh.multiply(4).divide(vv.add(vh)).rename("RVI")
        return img.addBands(rvi)

    collection = collection.map(add_sar_indices)
    sar_bands = ["VH", "VV", "RVI"]

    def extract_means(image):
        means = image.select(sar_bands).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry, scale=scale, maxPixels=1e9,
        )
        date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        return ee.Feature(None, means.set("date", date))

    all_features = []
    for b_start, b_end in _month_batches(start_date, end_date):
        try:
            batch = collection.filterDate(b_start, b_end).map(extract_means).getInfo()["features"]
            all_features.extend(batch)
        except Exception as exc:
            print(f"[collect_sentinel1] batch {b_start}–{b_end} failed: {exc}")

    rows = []
    for f in all_features:
        p = f["properties"]
        if p.get("VH") is not None:
            rows.append({
                "date": p["date"],
                "VH": round(p.get("VH") or 0, 4),
                "VV": round(p.get("VV") or 0, 4),
                "RVI": round(p.get("RVI") or 0, 4),
            })

    if not rows:
        return pd.DataFrame(columns=["date", "VH", "VV", "RVI"])

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df
