import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from lightgbm import LGBMRanker 
from lightgbm import early_stopping
from scipy.sparse import csr_matrix
import implicit

events = pd.read_csv("events.csv")
items1 = pd.read_csv("item_properties_part1.csv")
items2 = pd.read_csv("item_properties_part2.csv")
categories= pd.read_csv("category_tree.csv")

# items1 ve items2 data framelerini birleştir
item_properties = pd.concat([items1, items2], ignore_index=True)
print(item_properties.shape)
print(item_properties['property'].value_counts().head())
print(item_properties.head())

print(events['timestamp'].min())
print(events['timestamp'].max())

# Timestamp sütununu datetime çevir
events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')
item_properties['timestamp'] = pd.to_datetime(item_properties['timestamp'], unit='ms')

# Bir ürünün aynı özelliği birden fazla kez güncellenmiş olabilir, en son güncellenilen tarihi tut.
item_properties.sort_values("timestamp", ascending=True, inplace=True)
item_latest = item_properties.drop_duplicates(subset=["itemid", "property"], keep="last")

# Events ile item_category'yi birleştirelim(ama sadece category id olan sütunları aldık)
item_category = item_latest[item_latest['property'] == 'categoryid'][['itemid', 'value']].drop_duplicates(subset='itemid')
item_category = item_category.rename(columns={'value': 'categoryid'})

# Merge öncesi aynı itemid birden fazla mı?
print("item_category:", item_category['itemid'].duplicated().sum())

events_merged = pd.merge(events, item_category, on='itemid', how='left')
print(events_merged.head())

#item_propertiesdeki property sütununda yazan available kısmını aldık
item_available = item_latest[item_latest['property'] == 'available'][['itemid', 'value']].drop_duplicates(subset='itemid')
item_available = item_available.rename(columns={'value': 'isAvailable'})

print("item_available:", item_available['itemid'].duplicated().sum())

events_merged = pd.merge(events_merged, item_available, on='itemid', how='left')
print(events_merged.head())

# categoryid sütunundaki verileri numeric bir tipe çevir
events_merged['categoryid'] = pd.to_numeric(events_merged['categoryid'], errors='coerce')

# events_merged DataFrame'indeki 'categoryid' ile category_tree'yi birleştir.
events_final = pd.merge(events_merged, categories, on='categoryid', how='left')
events_final = events_final.rename(columns={'parentid': 'parent_categoryid'}) # Sütun adını değiştir

print(events_final.isnull().sum())

# isAvailable sütunundaki eksik değerleri 0 ile doldur. Bilinmiyorsa stok yokmuş gibi davranır.
events_final['isAvailable'] = events_final['isAvailable'].fillna(0).astype(int)
print(events_final['isAvailable'].value_counts(dropna=False)) 

# parent_id sütunundaki eksik değerleri -1 ile doldur
events_final['categoryid'] = events_final['categoryid'].fillna(-1).astype(int)
print(events_final['categoryid'].value_counts(dropna=False).head())

# categoryid'si -1 olanlar için is_unknown_category sütunu oluştur (1: bilinmeyen, 0: bilinen)
events_final['is_unknown_category'] = (events_final['categoryid'] == -1).astype(int)
print(events_final['is_unknown_category'].value_counts(dropna=False))

# parent_categoryid sütunundaki eksik değerleri -1 ile doldur
events_final['parent_categoryid'] = events_final['parent_categoryid'].fillna(-1).astype(int)
print(events_final['parent_categoryid'].value_counts(dropna=False).head()) # Kontrol edelim

# parent_categoryid'si -1 olanlar için is_unknown_parent_category sütunu oluştur
events_final['is_unknown_parent_category'] = (events_final['parent_categoryid'] == -1).astype(int)
print(events_final['is_unknown_parent_category'].value_counts(dropna=False))

# transactionid sütunundaki NaN olmayan değerler için 1, NaN olanlar için 0 atar.
# Bu, bir olayın bir satın alma işlemi olup olmadığını gösterir.
events_final['has_transaction'] = events_final['transactionid'].notna().astype(int)
print(events_final['has_transaction'].value_counts(dropna=False))
print(events_final.head())

# interaction_strength sütununu oluştur
event_weights = {'view': 1, 'addtocart': 8, 'transaction': 10}
events_final['interaction_strength'] = events_final['event'].map(event_weights)
print("events_final ilk 5 satırı:\n", events_final.head())
print("events_final Null değerler:\n", events_final.isnull().sum())

# 6. Bellek Optimizasyonu (events_final üzerinde) 
print("\n--- events_final Üzerinde Bellek Optimizasyonu Başlıyor ---")
def optimize_dataframe_memory(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'int64':
            max_val = df[col].max()
            min_val = df[col].min()
            if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif col_type == 'object':
            if pd.api.types.is_datetime64_any_dtype(df[col]): # Datetime sütunlarını atla
                continue
            # interaction_strength sütununu category'ye dönüştürmesini engelle
            if col == 'interaction_strength': # BURADA EKLENEN KONTROL
                continue
            num_unique = df[col].nunique()
            num_rows = len(df[col])
            if num_unique / num_rows < 0.5 and num_unique < 50000:
                df[col] = df[col].astype('category')
        elif col_type == 'float64':
            df[col] = df[col].astype(np.float32)
    return df

events_final = optimize_dataframe_memory(events_final)
print("--- events_final Bellek Optimizasyonu Tamamlandı ---")
print("events_final info (optimiation after merge):\n", events_final.info(memory_usage='deep'))

# Zaman Tabanlı Filtreleme
end_date = events_final['timestamp'].max()
start_date = end_date - pd.Timedelta(days=90) # Son 90 günü baz alıyor
recent_events = events_final[(events_final['timestamp'] >= start_date) & (events_final['timestamp'] <= end_date)].copy()
print(f"Son 90 gündeki olay sayısı (recent_events): {len(recent_events)}")

# split: train / val / test
test_end_date = recent_events['timestamp'].max()
test_start_date = test_end_date - pd.Timedelta(days=15)
validation_start_date = test_start_date - pd.Timedelta(days=15)

train_events = recent_events[recent_events['timestamp'] < validation_start_date].copy()
val_events = recent_events[(recent_events['timestamp'] >= validation_start_date) & (recent_events['timestamp'] < test_start_date)].copy()
test_events = recent_events[recent_events['timestamp'] >= test_start_date].copy()

# Kullanıcı Bazlı Feature Engineering
def add_user_features(df):
    # user_total_events
    user_total_events = df.groupby('visitorid').size().reset_index(name='user_total_events')
    df = pd.merge(df, user_total_events, on='visitorid', how='left')

    # user_view_count, user_addtocart_count, user_transaction_count
    user_event_counts = df.groupby(['visitorid', 'event']).size().unstack(fill_value=0).reset_index()
    user_event_counts.rename(columns={
        'view': 'user_view_count',
        'addtocart': 'user_addtocart_count',
        'transaction': 'user_transaction_count'
    }, inplace=True)
    df = pd.merge(df, user_event_counts, on='visitorid', how='left')

    # user_unique_categories
    user_unique_categories = df.groupby('visitorid')['categoryid'].nunique().reset_index(name='user_unique_categories')
    df = pd.merge(df, user_unique_categories, on='visitorid', how='left')

    # user_unique_items
    user_unique_items = df.groupby('visitorid')['itemid'].nunique().reset_index(name='user_unique_items')
    df = pd.merge(df, user_unique_items, on='visitorid', how='left')

    # user_conversion_rate
    df['user_conversion_rate'] = (df['user_transaction_count'] / df['user_view_count']).fillna(0)
    df.loc[(df['user_view_count'] == 0) & (df['user_transaction_count'] > 0), 'user_conversion_rate'] = 1

    # user_avg_interaction_strength
    user_avg_strength = df.groupby('visitorid')['interaction_strength'].mean().reset_index(name='user_avg_interaction_strength')
    df = pd.merge(df, user_avg_strength, on='visitorid', how='left')

    # user_unique_parent_categories
    unique_parent_cat = df.groupby('visitorid')['parent_categoryid'].nunique().reset_index(name='user_unique_parent_categories')
    df = pd.merge(df, unique_parent_cat, on='visitorid', how='left')

     # user_item_view_count
    user_item_view_count = df[df['event'] == 'view'].groupby(['visitorid', 'itemid']).size().reset_index(name='user_item_view_count')
    df = pd.merge(df, user_item_view_count, on=['visitorid', 'itemid'], how='left')

    # user_item_addtocart_count
    user_item_addtocart_count = df[df['event'] == 'addtocart'].groupby(['visitorid', 'itemid']).size().reset_index(name='user_item_addtocart_count')
    df = pd.merge(df, user_item_addtocart_count, on=['visitorid', 'itemid'], how='left')

    # user_item_interaction_strength_sum
    interaction_sum = df.groupby(['visitorid', 'itemid'])['interaction_strength'].sum().reset_index(name='user_item_interaction_strength_sum')
    df = pd.merge(df, interaction_sum, on=['visitorid', 'itemid'], how='left')

    # user_item_last_interaction_days_ago
    last_interaction = df.groupby(['visitorid', 'itemid'])['timestamp'].max().reset_index(name='user_item_last_interaction_date')
    end_date = df['timestamp'].max()
    last_interaction['user_item_last_interaction_days_ago'] = (end_date - last_interaction['user_item_last_interaction_date']).dt.days
    df = pd.merge(df, last_interaction[['visitorid', 'itemid', 'user_item_last_interaction_days_ago']], on=['visitorid', 'itemid'], how='left')
    
    return df

# Ürün Bazlı Feature Engineering
def add_item_features(df):
    # item_view_count, addtocart, transaction
    item_event_counts = df.groupby(['itemid', 'event']).size().unstack(fill_value=0).reset_index()
    item_event_counts.rename(columns={
        'view': 'item_view_count',
        'addtocart': 'item_addtocart_count',
        'transaction': 'item_transaction_count'
    }, inplace=True)
    df = pd.merge(df, item_event_counts, on='itemid', how='left')

    # item_ctr
    df['item_ctr'] = (df['item_transaction_count'] / df['item_view_count']).fillna(0)
    df.loc[(df['item_view_count'] == 0) & (df['item_transaction_count'] > 0), 'item_ctr'] = 1

    # item_unique_visitors
    unique_visitors = df.groupby('itemid')['visitorid'].nunique().reset_index(name='item_unique_visitors')
    df = pd.merge(df, unique_visitors, on='itemid', how='left')

    # item_avg_interaction_strength
    avg_strength = df.groupby('itemid')['interaction_strength'].mean().reset_index(name='item_avg_interaction_strength')
    df = pd.merge(df, avg_strength, on='itemid', how='left')

    # parent_category_view_count
    parent_view_count = df[df['event'] == 'view'].groupby('parent_categoryid').size().reset_index(name='parent_category_view_count')
    df = pd.merge(df, parent_view_count, on='parent_categoryid', how='left')

    # parent_category_transaction_count
    parent_trans_count = df[df['event'] == 'transaction'].groupby('parent_categoryid').size().reset_index(name='parent_category_transaction_count')
    df = pd.merge(df, parent_trans_count, on='parent_categoryid', how='left')

    return df

# Yer kaplayan data setlerini silme işlemi:
del items1, items2, events, events_merged, item_properties, item_available, item_category, events_final

# train_events için özellik oluşturma
train_events = add_user_features(train_events)
train_events = add_item_features(train_events)

# val_events için özellik oluşturma
val_events = add_user_features(val_events)
val_events = add_item_features(val_events)

# test_events için özellik oluşturma
test_events = add_user_features(test_events)
test_events = add_item_features(test_events)

# Aykırı değer var mı kontrolü:
def find_outliers_iqr(df, column_name):

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    # 1.5 * IQR kuralını kullanarak üst ve alt limitleri belirleme
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    
    # Aykırı değerleri filtreleme
    outliers = df[(df[column_name] > upper_limit) | (df[column_name] < lower_limit)]
    
    print(f"'{column_name}' sütununda {len(outliers)} adet aykırı değer bulundu.")
    print("-" * 50)
    if not outliers.empty:
        print("İlk 5 aykırı değer örneği:")
        print(outliers[['visitorid', column_name]].head())
    else:
        print("Aykırı değer bulunamadı.")
    
    return outliers

for df in [train_events, val_events, test_events]:
    outliers_user_events = find_outliers_iqr(df, 'user_total_events')
    outliers_item_views = find_outliers_iqr(df, 'item_view_count')
    outliers_item_ctr = find_outliers_iqr(df, 'item_ctr')

# --- Log dönüşümü uygulanacak sütunları otomatik belirleme ---
def get_log_columns(df):
    log_cols = []
    for col in df.columns:
        # sayısal olmalı
        if pd.api.types.is_numeric_dtype(df[col]):
            # kategorik id veya binary olmamalı
            if col not in ['categoryid', 'parent_categoryid', 'isAvailable']:
                # sayısal ama aynı zamanda büyük ve dengesiz dağılıma sahip olma ihtimali yüksek
                if '_count' in col or '_total_events' in col or '_unique_' in col or '_avg_' in col:
                    log_cols.append(col)
    return log_cols

log_cols_user = get_log_columns(train_events[[c for c in train_events.columns if c.startswith('user_')]])
log_cols_item = get_log_columns(train_events[[c for c in train_events.columns if c.startswith('item_') or c.startswith('parent_')]])
log_cols_all = log_cols_user + log_cols_item + ['interaction_strength']


# Log dönüşümünü her bir veri seti için ayrı ayrı uygulayın
for col in log_cols_all:
    if col in train_events.columns:
        train_events[f'{col}_log'] = np.log1p(train_events[col])
    if col in val_events.columns:
        val_events[f'{col}_log'] = np.log1p(val_events[col])
    if col in test_events.columns:
        test_events[f'{col}_log'] = np.log1p(test_events[col])

# --- 3. ALS Matrisini Oluşturma (train_events ve val_events ile) ---
print("\nALS matrisi, eğitim ve validasyon verileri birleştirilerek oluşturuluyor...")

# Eğitim ve validasyon setlerini birleştirme
combined_events = pd.concat([train_events, val_events], ignore_index=True)

# Eşlemeleri (mappings) birleşik veriden oluşturma
visitor_to_idx = {v: i for i, v in enumerate(combined_events['visitorid'].unique())}
item_to_idx = {item: idx for idx, item in enumerate(combined_events['itemid'].unique())}
idx_to_item = {idx: item for item, idx in item_to_idx.items()}

# Birleştirilmiş verilere visitor_idx ve item_idx ekleme
combined_events['visitor_idx'] = combined_events['visitorid'].map(visitor_to_idx)
combined_events['item_idx'] = combined_events['itemid'].map(item_to_idx)

# Matris oluşturma
user_item_matrix = csr_matrix(
    (
        combined_events['interaction_strength'].astype('float32'),  # interaction_strength_log yerine orijinal değer
        (combined_events['visitor_idx'], combined_events['item_idx'])
    ),
    shape=(len(visitor_to_idx), len(item_to_idx))
)

# --- 4. ALS Modelini Eğitme ---
print("\nALS modeli, birleştirilmiş veri ile eğitiliyor...")

model_als = implicit.als.AlternatingLeastSquares(
    factors=64,
    regularization=0.05,
    iterations=50,
    random_state=42
)

# Eğitimi başlatma
model_als.fit(user_item_matrix.T.tocsr())
print("ALS modeli eğitimi tamamlandı.")

# --- 5. LightGBM İçin Özellikleri SADECE eğitim verisinden oluştur ---
print("\nLightGBM için özellikler sadece eğitim verisinden türetiliyor...")

user_features = train_events.drop_duplicates(subset='visitorid').copy()
item_features = train_events.drop_duplicates(subset='itemid').copy()

# Logaritmik ve diğer özellikleri ayırma

# Kullanıcı (user) özellikleri
user_features_log = user_features[
    ['visitorid'] + [col for col in user_features.columns if 'user_' in col and 'log' in col]
].drop_duplicates(subset=['visitorid'])

# Kullanıcı log özelliklerinden user_item_* log kolonlarını çıkar
user_features_log = user_features_log[[c for c in user_features_log.columns if not c.startswith('user_item_')]]

user_features_no_log = user_features[
    ['visitorid'] + [col for col in user_features.columns if 'user_' in col and 'log' not in col]
].drop_duplicates(subset=['visitorid'])

# Ürün (item) özellikleri
item_features_log = item_features[
    ['itemid'] + [col for col in item_features.columns if 'item_' in col and 'log' in col]
].drop_duplicates(subset=['itemid'])

# Ürün log özelliklerinden user_item_* log kolonlarını çıkar (tedbiren)
item_features_log = item_features_log[[c for c in item_features_log.columns if not c.startswith('user_item_')]]

item_features_no_log = item_features[
    ['itemid'] + [
        col for col in item_features.columns
        if ('log' not in col) and (
            col.startswith('item_') or
            col.startswith('parent_') or
            col in ['isAvailable', 'categoryid', 'parent_categoryid']
        )
    ]
].drop_duplicates(subset=['itemid'])

# --- 6. LightGBM Eğitim ve Validasyon Setlerini Birleştirme ---
print("\nLightGBM eğitim ve validasyon setleri hazırlanıyor...")

# Eğitim setini oluştur
train_set = train_events[['visitorid', 'itemid', 'has_transaction']].drop_duplicates().copy()
train_set.rename(columns={'has_transaction': 'target'}, inplace=True)
train_set.fillna(0, inplace=True)

# Validasyon setini oluştur
val_set = val_events[['visitorid', 'itemid', 'has_transaction']].drop_duplicates().copy()
val_set.rename(columns={'has_transaction': 'target'}, inplace=True)
val_set.fillna(0, inplace=True)

# Özellikleri birleştir
train_set = pd.merge(train_set, user_features_no_log, on='visitorid', how='left')
train_set = pd.merge(train_set, user_features_log, on='visitorid', how='left')
train_set = pd.merge(train_set, item_features_no_log, on='itemid', how='left')
train_set = pd.merge(train_set, item_features_log, on='itemid', how='left')
train_set.fillna(0, inplace=True)

val_set = pd.merge(val_set, user_features_no_log, on='visitorid', how='left')
val_set = pd.merge(val_set, user_features_log, on='visitorid', how='left')
val_set = pd.merge(val_set, item_features_no_log, on='itemid', how='left')
val_set = pd.merge(val_set, item_features_log, on='itemid', how='left')
val_set.fillna(0, inplace=True)

# x_y ile biten sütunları listele
x_y_columns = [col for col in train_set.columns if col.endswith('_x') or col.endswith('_y')]
print("Çakışan sütunlar:", x_y_columns)

# x_y ile biten sütunları listele
x_y_columns = [col for col in val_set.columns if col.endswith('_x') or col.endswith('_y')]
print("Çakışan sütunlar:", x_y_columns)

# Kategorik özellikleri dönüştürme
for col in ['isAvailable', 'categoryid', 'parent_categoryid']:
    if col in train_set.columns:
        train_set[col] = train_set[col].astype('category')
        val_set[col] = val_set[col].astype('category')

# Features listesini oluşturma
features = [col for col in train_set.columns if col not in ['visitorid', 'itemid', 'target', 'timestamp', 'event', 'transactionid', 'visitor_idx', 'item_idx']]

X_train, y_train = train_set[features], train_set['target']
X_val, y_val = val_set[features], val_set['target']

# Gruplama için `groupby` kullanma
train_group = train_set.groupby('visitorid').size().values
val_group = val_set.groupby('visitorid').size().values

n_pos = len(train_set[train_set['target'] == 1])
n_neg = len(train_set[train_set['target'] == 0])

scale_pos_weight = n_neg / n_pos
print(f"Pozitif: {n_pos}, Negatif: {n_neg}, scale_pos_weight: {scale_pos_weight:.2f}")

# --- 7. LightGBM Ranker Model Eğitimi ---
print("\nLightGBM Ranker modeli eğitiliyor...")
lgb_ranker = LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    random_state=42,
    n_estimators=500,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight
)

lgb_ranker.fit(X_train, y_train,
              group=train_group,
              eval_set=[(X_val, y_val)],
              eval_group=[val_group],
              eval_metric='ndcg',
              callbacks=[early_stopping(100, verbose=False)])

print("LightGBM Ranker eğitimi tamamlandı.")

features_lgbm = X_train.columns.tolist()

user_features_full = train_events[
    ['visitorid'] + [col for col in train_events.columns if col.startswith('user_')]
].drop_duplicates(subset='visitorid')

item_features_full = train_events[
    ['itemid'] + [
        col for col in train_events.columns
        if col.startswith('item_') or col.startswith('parent_') or 
           col in ['isAvailable', 'categoryid', 'parent_categoryid']
    ]
].drop_duplicates(subset='itemid')

def calculate_recall_at_k(model_lgbm, model_als, train_events, test_events,
                          visitor_to_idx, item_to_idx, user_item_matrix, item_features, user_features,
                          features_lgbm, k=10, top_k_als=300):

    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    recall_scores = []
    test_visitors_with_transactions = test_events[test_events['has_transaction'] == 1]['visitorid'].unique()

    for visitor_id in test_visitors_with_transactions:
        # 1. ALS ile aday üretimi
        candidate_items_als = []
        if visitor_id in visitor_to_idx:
            user_idx = visitor_to_idx[visitor_id]

            # EK KONTROL: user_idx'in modelin boyutları içinde olup olmadığını kontrol et
            if user_idx >= model_als.user_factors.shape[0]:
                # Bu durumda kullanıcıyı atla, çünkü ALS modeli bu indeksi tanımıyor.
                continue

            # Sadece o kullanıcıya ait matris satırını al
            user_item_row = user_item_matrix.getrow(user_idx)
            als_recommendations = model_als.recommend(user_idx, user_item_row, N=top_k_als)

            # Güvenli çeviri: idx_to_item olmayanları atla
            for rec_tuple in als_recommendations:
                item_idx = rec_tuple[0]
                if item_idx in idx_to_item:
                    candidate_items_als.append(idx_to_item[item_idx])

        else:
            # Cold-start kullanıcısı: En popüler ürünleri aday olarak al
            candidate_items_als = item_features.sort_values('item_transaction_count', ascending=False)['itemid'].head(top_k_als).tolist()

        # Eğitimde görülen ürünleri önerilerden çıkar
        seen_items = set(train_events[train_events['visitorid'] == visitor_id]['itemid'])
        candidate_items_als = [item for item in candidate_items_als if item not in seen_items]

        if not candidate_items_als:
            continue

        # 2. LightGBM için özelliklerle DataFrame oluştur
        candidate_df = pd.DataFrame({'visitorid': [visitor_id] * len(candidate_items_als), 'itemid': candidate_items_als})
        candidate_df = pd.merge(candidate_df, user_features, on='visitorid', how='left')
        candidate_df = pd.merge(candidate_df, item_features, on='itemid', how='left')
        candidate_df.fillna(0, inplace=True)

        # Kategorik özellikleri dönüştür (eğer varsa)
        for col in ['isAvailable', 'categoryid', 'parent_categoryid']:
            if col in candidate_df.columns:
                candidate_df[col] = candidate_df[col].astype('category')

        # Modelin beklediği tüm feature'ların DataFrame'de olduğundan emin ol
        for col in features_lgbm:
            if col not in candidate_df.columns:
                candidate_df[col] = 0

        # 3. LightGBM ile sıralama yap
        candidate_features = candidate_df[features_lgbm]
        if candidate_features.empty:
            continue

        candidate_df['score'] = model_lgbm.predict(candidate_features)

        # En iyi k ürünü seç
        top_k_predicted_items = candidate_df.sort_values('score', ascending=False)['itemid'].head(k).tolist()

        # 4. Recall hesaplaması (sadece işlem yapılan ürünler)
        actual_purchased_items = test_events[(test_events['visitorid'] == visitor_id) & (test_events['has_transaction'] == 1)]['itemid'].unique()

        if len(actual_purchased_items) > 0:
            hits = len(set(top_k_predicted_items) & set(actual_purchased_items))
            recall_at_user = hits / len(actual_purchased_items)
            recall_scores.append(recall_at_user)

    return np.mean(recall_scores) if recall_scores else 0.0

# --- Recall@10'u hesapla ---
#features_lgbm = lgb_ranker.feature_name_

recall_10 = calculate_recall_at_k(
    lgb_ranker, model_als,
    train_events, test_events,
    visitor_to_idx, item_to_idx,
    user_item_matrix, item_features_full, 
    user_features_full, features_lgbm,
    k=10, top_k_als=300)
print(f"Hibrit Model için Recall@{10}: {recall_10:.4f}")

# Özellik isimlerini ve önem skorlarını al
feature_importances = pd.DataFrame({
    'feature': lgb_ranker.feature_name_,
    'importance': lgb_ranker.feature_importances_
}).sort_values(by='importance', ascending=False)

# İlk 20 en önemli özelliği seç ve görselleştir
top_20_features = feature_importances.head(20)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=top_20_features, palette='viridis')
plt.title('Top 20 LightGBM Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

feature_scores = pd.DataFrame({
    'feature': lgb_ranker.feature_name_,
    'importance': lgb_ranker.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_scores.head(20))