import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';
import { Box, Typography, useTheme } from '@mui/material';
import { format, parseISO } from 'date-fns';

const ThreatTimeline = ({ data, timeRange }) => {
  const theme = useTheme();

  const processedData = useMemo(() => {
    if (!data || !Array.isArray(data)) return [];

    return data.map(item => ({
      ...item,
      timestamp: new Date(item.timestamp).getTime(),
      total: (item.critical || 0) + (item.high || 0) + (item.medium || 0) + (item.low || 0),
    }));
  }, [data]);

  const formatXAxisLabel = (tickItem) => {
    const date = new Date(tickItem);
    switch (timeRange) {
      case '1h':
        return format(date, 'HH:mm');
      case '6h':
        return format(date, 'HH:mm');
      case '24h':
        return format(date, 'HH:mm');
      case '7d':
        return format(date, 'MM/dd');
      default:
        return format(date, 'HH:mm');
    }
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const date = new Date(label);
      return (
        <Box
          sx={{
            bgcolor: 'background.paper',
            p: 2,
            border: 1,
            borderColor: 'divider',
            borderRadius: 1,
            boxShadow: 2,
          }}
        >
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            {format(date, 'MMM dd, yyyy HH:mm')}
          </Typography>
          {payload.map((entry, index) => (
            <Box key={index} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  bgcolor: entry.color,
                  borderRadius: '50%',
                  mr: 1,
                }}
              />
              <Typography variant="body2">
                {entry.name}: {entry.value}
              </Typography>
            </Box>
          ))}
          <Box sx={{ mt: 1, pt: 1, borderTop: 1, borderColor: 'divider' }}>
            <Typography variant="body2" fontWeight="bold">
              Total: {payload.reduce((sum, entry) => sum + entry.value, 0)}
            </Typography>
          </Box>
        </Box>
      );
    }
    return null;
  };

  if (!processedData.length) {
    return (
      <Box
        sx={{
          height: 300,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Typography variant="body2" color="text.secondary">
          No threat data available for the selected time range
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', height: 300 }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={processedData}
          margin={{
            top: 10,
            right: 30,
            left: 0,
            bottom: 0,
          }}
        >
          <defs>
            <linearGradient id="criticalGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={theme.palette.error.main} stopOpacity={0.8} />
              <stop offset="95%" stopColor={theme.palette.error.main} stopOpacity={0.1} />
            </linearGradient>
            <linearGradient id="highGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={theme.palette.warning.main} stopOpacity={0.8} />
              <stop offset="95%" stopColor={theme.palette.warning.main} stopOpacity={0.1} />
            </linearGradient>
            <linearGradient id="mediumGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={theme.palette.info.main} stopOpacity={0.8} />
              <stop offset="95%" stopColor={theme.palette.info.main} stopOpacity={0.1} />
            </linearGradient>
            <linearGradient id="lowGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={theme.palette.success.main} stopOpacity={0.8} />
              <stop offset="95%" stopColor={theme.palette.success.main} stopOpacity={0.1} />
            </linearGradient>
          </defs>
          
          <CartesianGrid 
            strokeDasharray="3 3" 
            stroke={theme.palette.divider}
            opacity={0.5}
          />
          
          <XAxis
            dataKey="timestamp"
            type="number"
            scale="time"
            domain={['dataMin', 'dataMax']}
            tickFormatter={formatXAxisLabel}
            stroke={theme.palette.text.secondary}
            fontSize={12}
          />
          
          <YAxis
            stroke={theme.palette.text.secondary}
            fontSize={12}
          />
          
          <Tooltip content={<CustomTooltip />} />
          
          <Legend
            wrapperStyle={{
              paddingTop: '20px',
              fontSize: '12px',
            }}
          />

          <Area
            type="monotone"
            dataKey="critical"
            stackId="1"
            stroke={theme.palette.error.main}
            fill="url(#criticalGradient)"
            name="Critical"
          />
          
          <Area
            type="monotone"
            dataKey="high"
            stackId="1"
            stroke={theme.palette.warning.main}
            fill="url(#highGradient)"
            name="High"
          />
          
          <Area
            type="monotone"
            dataKey="medium"
            stackId="1"
            stroke={theme.palette.info.main}
            fill="url(#mediumGradient)"
            name="Medium"
          />
          
          <Area
            type="monotone"
            dataKey="low"
            stackId="1"
            stroke={theme.palette.success.main}
            fill="url(#lowGradient)"
            name="Low"
          />

          {/* Add reference line for average */}
          {processedData.length > 0 && (
            <ReferenceLine
              y={processedData.reduce((sum, item) => sum + item.total, 0) / processedData.length}
              stroke={theme.palette.text.secondary}
              strokeDasharray="5 5"
              label="Average"
            />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default ThreatTimeline;