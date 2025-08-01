import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  IconButton,
  Tooltip,
  Alert,
  Fade,
  Zoom,
} from '@mui/material';
import {
  Security,
  Warning,
  TrendingUp,
  TrendingDown,
  Refresh,
  FilterList,
  Download,
  Visibility,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';

// Components
import MetricCard from '../components/Dashboard/MetricCard';
import ThreatTimeline from '../components/Dashboard/ThreatTimeline';
import ThreatDistribution from '../components/Dashboard/ThreatDistribution';
import RecentAlerts from '../components/Dashboard/RecentAlerts';
import ThreatMap from '../components/Dashboard/ThreatMap';
import SystemHealth from '../components/Dashboard/SystemHealth';

// Hooks
import { useQuery } from 'react-query';
import { useThreatData } from '../hooks/useThreatData';
import { useRealTimeUpdates } from '../hooks/useRealTimeUpdates';

const ThreatOverview = () => {
  const [timeRange, setTimeRange] = useState('24h');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedSeverity, setSelectedSeverity] = useState('all');

  // Fetch threat data
  const {
    data: threatData,
    isLoading,
    error,
    refetch,
  } = useThreatData(timeRange, selectedSeverity);

  // Real-time updates
  const { lastUpdate, isConnected } = useRealTimeUpdates();

  // Auto-refresh effect
  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        refetch();
      }, 30000); // Refresh every 30 seconds

      return () => clearInterval(interval);
    }
  }, [autoRefresh, refetch]);

  const handleRefresh = () => {
    refetch();
  };

  const handleTimeRangeChange = (newRange) => {
    setTimeRange(newRange);
  };

  const handleSeverityFilter = (severity) => {
    setSelectedSeverity(severity);
  };

  if (isLoading) {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <LinearProgress />
        <Typography variant="h6" sx={{ mt: 2, textAlign: 'center' }}>
          Loading threat data...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        Failed to load threat data. Please try again.
      </Alert>
    );
  }

  const {
    totalThreats = 0,
    criticalThreats = 0,
    activeIncidents = 0,
    resolvedThreats = 0,
    threatTrend = 0,
    systemHealth = 'good',
    recentAlerts = [],
    threatDistribution = [],
    timelineData = [],
    geographicData = [],
  } = threatData || {};

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Box sx={{ mb: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" component="h1" sx={{ fontWeight: 'bold' }}>
            Threat Overview
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Tooltip title="Refresh Data">
              <IconButton onClick={handleRefresh} color="primary">
                <Refresh />
              </IconButton>
            </Tooltip>
            <Tooltip title="Filter Options">
              <IconButton color="primary">
                <FilterList />
              </IconButton>
            </Tooltip>
            <Tooltip title="Export Data">
              <IconButton color="primary">
                <Download />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Connection Status */}
        <AnimatePresence>
          {!isConnected && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
            >
              <Alert severity="warning" sx={{ mb: 2 }}>
                Real-time connection lost. Data may not be current.
              </Alert>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Key Metrics */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              title="Total Threats"
              value={totalThreats}
              trend={threatTrend}
              icon={<Security />}
              color="primary"
              subtitle="Last 24 hours"
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              title="Critical Threats"
              value={criticalThreats}
              trend={-2}
              icon={<Warning />}
              color="error"
              subtitle="Requires immediate attention"
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              title="Active Incidents"
              value={activeIncidents}
              trend={1}
              icon={<TrendingUp />}
              color="warning"
              subtitle="Currently investigating"
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              title="Resolved"
              value={resolvedThreats}
              trend={5}
              icon={<TrendingDown />}
              color="success"
              subtitle="Resolution rate: 94%"
            />
          </Grid>
        </Grid>

        {/* Main Dashboard Grid */}
        <Grid container spacing={3}>
          {/* Threat Timeline */}
          <Grid item xs={12} lg={8}>
            <Card sx={{ height: 400 }}>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" component="h2">
                    Threat Activity Timeline
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    {['1h', '6h', '24h', '7d'].map((range) => (
                      <Chip
                        key={range}
                        label={range}
                        onClick={() => handleTimeRangeChange(range)}
                        color={timeRange === range ? 'primary' : 'default'}
                        size="small"
                        clickable
                      />
                    ))}
                  </Box>
                </Box>
                <ThreatTimeline data={timelineData} timeRange={timeRange} />
              </CardContent>
            </Card>
          </Grid>

          {/* System Health */}
          <Grid item xs={12} lg={4}>
            <Card sx={{ height: 400 }}>
              <CardContent>
                <Typography variant="h6" component="h2" sx={{ mb: 2 }}>
                  System Health
                </Typography>
                <SystemHealth status={systemHealth} />
              </CardContent>
            </Card>
          </Grid>

          {/* Threat Distribution */}
          <Grid item xs={12} md={6}>
            <Card sx={{ height: 350 }}>
              <CardContent>
                <Typography variant="h6" component="h2" sx={{ mb: 2 }}>
                  Attack Types Distribution
                </Typography>
                <ThreatDistribution data={threatDistribution} />
              </CardContent>
            </Card>
          </Grid>

          {/* Geographic Threat Map */}
          <Grid item xs={12} md={6}>
            <Card sx={{ height: 350 }}>
              <CardContent>
                <Typography variant="h6" component="h2" sx={{ mb: 2 }}>
                  Geographic Threat Distribution
                </Typography>
                <ThreatMap data={geographicData} />
              </CardContent>
            </Card>
          </Grid>

          {/* Recent Alerts */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" component="h2">
                    Recent Security Alerts
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    {['all', 'critical', 'high', 'medium', 'low'].map((severity) => (
                      <Chip
                        key={severity}
                        label={severity.charAt(0).toUpperCase() + severity.slice(1)}
                        onClick={() => handleSeverityFilter(severity)}
                        color={selectedSeverity === severity ? 'primary' : 'default'}
                        size="small"
                        clickable
                      />
                    ))}
                  </Box>
                </Box>
                <RecentAlerts 
                  alerts={recentAlerts} 
                  onAlertClick={(alert) => console.log('Alert clicked:', alert)}
                />
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Real-time Status Bar */}
        <Box
          sx={{
            position: 'fixed',
            bottom: 16,
            right: 16,
            zIndex: 1000,
          }}
        >
          <Fade in={isConnected}>
            <Card sx={{ bgcolor: 'success.main', color: 'success.contrastText' }}>
              <CardContent sx={{ py: 1, px: 2, '&:last-child': { pb: 1 } }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box
                    sx={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      bgcolor: 'success.contrastText',
                      animation: 'pulse 2s infinite',
                    }}
                  />
                  <Typography variant="caption">
                    Live • Last update: {lastUpdate}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Fade>
        </Box>
      </Box>

      <style jsx>{`
        @keyframes pulse {
          0% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
          100% {
            opacity: 1;
          }
        }
      `}</style>
    </motion.div>
  );
};

export default ThreatOverview;