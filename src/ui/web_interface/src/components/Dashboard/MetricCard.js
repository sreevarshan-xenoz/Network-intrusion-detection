import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  MoreVert,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

const MetricCard = ({
  title,
  value,
  trend,
  icon,
  color = 'primary',
  subtitle,
  onClick,
  loading = false,
}) => {
  const getTrendIcon = () => {
    if (trend > 0) return <TrendingUp sx={{ fontSize: 16 }} />;
    if (trend < 0) return <TrendingDown sx={{ fontSize: 16 }} />;
    return <TrendingFlat sx={{ fontSize: 16 }} />;
  };

  const getTrendColor = () => {
    if (trend > 0) return 'success';
    if (trend < 0) return 'error';
    return 'default';
  };

  const formatValue = (val) => {
    if (typeof val === 'number') {
      if (val >= 1000000) return `${(val / 1000000).toFixed(1)}M`;
      if (val >= 1000) return `${(val / 1000).toFixed(1)}K`;
      return val.toLocaleString();
    }
    return val;
  };

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      transition={{ type: 'spring', stiffness: 300 }}
    >
      <Card
        sx={{
          height: '100%',
          cursor: onClick ? 'pointer' : 'default',
          transition: 'all 0.3s ease',
          '&:hover': {
            boxShadow: (theme) => theme.shadows[8],
          },
        }}
        onClick={onClick}
      >
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <Box sx={{ flexGrow: 1 }}>
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ mb: 1, fontWeight: 500 }}
              >
                {title}
              </Typography>
              
              <Typography
                variant="h4"
                component="div"
                sx={{
                  fontWeight: 'bold',
                  color: `${color}.main`,
                  mb: 1,
                }}
              >
                {loading ? '...' : formatValue(value)}
              </Typography>

              {subtitle && (
                <Typography variant="caption" color="text.secondary">
                  {subtitle}
                </Typography>
              )}

              {trend !== undefined && (
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                  <Chip
                    icon={getTrendIcon()}
                    label={`${trend > 0 ? '+' : ''}${trend}`}
                    size="small"
                    color={getTrendColor()}
                    variant="outlined"
                    sx={{ fontSize: '0.75rem' }}
                  />
                </Box>
              )}
            </Box>

            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <Box
                sx={{
                  p: 1,
                  borderRadius: 2,
                  bgcolor: `${color}.light`,
                  color: `${color}.main`,
                  mb: 1,
                }}
              >
                {icon}
              </Box>
              
              <Tooltip title="More options">
                <IconButton size="small" sx={{ opacity: 0.7 }}>
                  <MoreVert fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default MetricCard;