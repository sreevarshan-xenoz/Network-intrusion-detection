// MongoDB initialization script for NIDS
db = db.getSiblingDB('nids');

// Create collections
db.createCollection('predictions');
db.createCollection('models');
db.createCollection('alerts');
db.createCollection('performance_metrics');

// Create indexes for better performance
db.predictions.createIndex({ "timestamp": 1 });
db.predictions.createIndex({ "source_ip": 1 });
db.predictions.createIndex({ "is_malicious": 1 });
db.predictions.createIndex({ "attack_type": 1 });
db.predictions.createIndex({ "model_version": 1 });

db.models.createIndex({ "model_id": 1 }, { unique: true });
db.models.createIndex({ "version": 1 });
db.models.createIndex({ "training_date": 1 });
db.models.createIndex({ "is_active": 1 });

db.alerts.createIndex({ "timestamp": 1 });
db.alerts.createIndex({ "severity": 1 });
db.alerts.createIndex({ "source_ip": 1 });
db.alerts.createIndex({ "attack_type": 1 });

db.performance_metrics.createIndex({ "timestamp": 1 });
db.performance_metrics.createIndex({ "metric_type": 1 });

// Create user for application
db.createUser({
  user: "nids_app",
  pwd: "nids_app_password",
  roles: [
    {
      role: "readWrite",
      db: "nids"
    }
  ]
});

print("MongoDB initialization completed for NIDS database");