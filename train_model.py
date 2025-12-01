# train_model.py
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import pickle
import os

def load_and_preprocess_data():
    """加载并预处理数据"""
    print("正在加载数据...")
    
    # 加载训练数据
    train_data = pd.read_csv("data/训练集smote.csv")
    
    # 加载验证数据
    test_data = pd.read_csv("data/验证集.csv")
    
    # 选择特征（基于您提供的变量列表）
    features = [
        "New_focal", "CDFI", "SIRI_four", "ETE", "TSH", 
        "Tumor_size_custom", "NG", "Boundary", "Microcalcification", "LMR_four"
    ]
    target = "HVCLNM"
    
    # 确保所有特征都存在
    available_features = [f for f in features if f in train_data.columns and f in test_data.columns]
    print(f"使用的特征: {available_features}")
    
    # 提取特征和目标
    X_train = train_data[available_features]
    y_train = train_data[target]
    X_test = test_data[available_features]
    y_test = test_data[target]
    
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"训练集HVCLNM分布:\n{y_train.value_counts()}")
    print(f"测试集HVCLNM分布:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test, available_features

def train_catboost_model(X_train, X_test, y_train, y_test, features):
    """训练CatBoost模型"""
    print("开始训练CatBoost模型...")
    
    # CatBoost参数（基于您之前的R代码参数）
    model = CatBoostClassifier(
        iterations=100,
        depth=5,
        learning_rate=0.03,
        random_seed=123,
        eval_metric='AUC',
        use_best_model=True,
        verbose=100
    )
    
    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        cat_features=[],  # 所有特征都是数值型
        plot=False,
        verbose=True
    )
    
    return model

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"测试集AUC: {auc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, auc

def save_model_and_features(model, features, accuracy, auc):
    """保存模型和特征信息"""
    # 创建models文件夹
    os.makedirs('models', exist_ok=True)
    
    # 保存模型
    with open('models/catboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # 保存特征信息
    model_info = {
        'features': features,
        'accuracy': accuracy,
        'auc': auc,
        'feature_importance': dict(zip(features, model.get_feature_importance()))
    }
    
    with open('models/model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print("模型和特征信息已保存!")
    print(f"特征重要性: {model_info['feature_importance']}")

def main():
    """主函数"""
    try:
        # 加载数据
        X_train, X_test, y_train, y_test, features = load_and_preprocess_data()
        
        # 训练模型
        model = train_catboost_model(X_train, X_test, y_train, y_test, features)
        
        # 评估模型
        accuracy, auc = evaluate_model(model, X_test, y_test)
        
        # 保存模型
        save_model_and_features(model, features, accuracy, auc)
        
        print("模型训练完成!")
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()