import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Badge,
  Avatar,
  Menu,
  MenuItem,
  Chip,
  Tooltip,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Notifications,
  AccountCircle,
  Brightness4,
  Brightness7,
  Refresh,
  Settings,
  ExitToApp,
  Circle,
} from '@mui/icons-material';

const Header = ({ onMenuClick, onThemeToggle, theme, isConnected }) => {
  const [anchorEl, setAnchorEl] = useState(null);
  const [notificationAnchor, setNotificationAnchor] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const handleProfileMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleProfileMenuClose = () => {
    setAnchorEl(null);
  };

  const handleNotificationMenuOpen = (event) => {
    setNotificationAnchor(event.currentTarget);
  };

  const handleNotificationMenuClose = () => {
    setNotificationAnchor(null);
  };

  const handleRefresh = () => {
    window.location.reload();
  };

  const notifications = [
    {
      id: 1,
      title: 'Critical Alert',
      message: 'DDoS attack detected from 192.168.1.100',
      time: '2 minutes ago',
      severity: 'critical',
    },
    {
      id: 2,
      title: 'New Threat Campaign',
      message: 'APT group activity identified',
      time: '15 minutes ago',
      severity: 'high',
    },
    {
      id: 3,
      title: 'System Update',
      message: 'Threat intelligence feeds updated',
      time: '1 hour ago',
      severity: 'info',
    },
  ];

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical':
        return 'error';
      case 'high':
        return 'warning';
      case 'medium':
        return 'info';
      case 'low':
        return 'success';
      default:
        return 'default';
    }
  };

  return (
    <AppBar
      position="sticky"
      sx={{
        zIndex: (theme) => theme.zIndex.drawer + 1,
        backgroundColor: 'background.paper',
        color: 'text.primary',
        boxShadow: 1,
        borderBottom: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Toolbar>
        {/* Menu Button */}
        <IconButton
          color="inherit"
          aria-label="open drawer"
          onClick={onMenuClick}
          edge="start"
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>

        {/* Title and Status */}
        <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center' }}>
          <Typography variant="h6" noWrap component="div" sx={{ mr: 2 }}>
            Network Security Dashboard
          </Typography>
          
          {/* Connection Status */}
          <Chip
            icon={
              <Circle
                sx={{
                  fontSize: 12,
                  color: isConnected ? 'success.main' : 'error.main',
                }}
              />
            }
            label={isConnected ? 'Connected' : 'Disconnected'}
            size="small"
            variant="outlined"
            sx={{ mr: 2 }}
          />

          {/* Last Update Time */}
          <Typography variant="caption" color="text.secondary">
            Last updated: {new Date().toLocaleTimeString()}
          </Typography>
        </Box>

        {/* Auto Refresh Toggle */}
        <FormControlLabel
          control={
            <Switch
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              size="small"
            />
          }
          label="Auto Refresh"
          sx={{ mr: 2 }}
        />

        {/* Refresh Button */}
        <Tooltip title="Refresh Dashboard">
          <IconButton color="inherit" onClick={handleRefresh} sx={{ mr: 1 }}>
            <Refresh />
          </IconButton>
        </Tooltip>

        {/* Theme Toggle */}
        <Tooltip title="Toggle Theme">
          <IconButton color="inherit" onClick={onThemeToggle} sx={{ mr: 1 }}>
            {theme === 'dark' ? <Brightness7 /> : <Brightness4 />}
          </IconButton>
        </Tooltip>

        {/* Notifications */}
        <Tooltip title="Notifications">
          <IconButton
            color="inherit"
            onClick={handleNotificationMenuOpen}
            sx={{ mr: 1 }}
          >
            <Badge badgeContent={notifications.length} color="error">
              <Notifications />
            </Badge>
          </IconButton>
        </Tooltip>

        {/* Profile Menu */}
        <Tooltip title="Account">
          <IconButton
            color="inherit"
            onClick={handleProfileMenuOpen}
            sx={{ ml: 1 }}
          >
            <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
              A
            </Avatar>
          </IconButton>
        </Tooltip>

        {/* Notification Menu */}
        <Menu
          anchorEl={notificationAnchor}
          open={Boolean(notificationAnchor)}
          onClose={handleNotificationMenuClose}
          PaperProps={{
            sx: { width: 350, maxHeight: 400 },
          }}
        >
          <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
            <Typography variant="h6">Notifications</Typography>
          </Box>
          {notifications.map((notification) => (
            <MenuItem
              key={notification.id}
              onClick={handleNotificationMenuClose}
              sx={{ whiteSpace: 'normal', py: 2 }}
            >
              <Box sx={{ width: '100%' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Typography variant="subtitle2" sx={{ flexGrow: 1 }}>
                    {notification.title}
                  </Typography>
                  <Chip
                    label={notification.severity}
                    size="small"
                    color={getSeverityColor(notification.severity)}
                    sx={{ ml: 1 }}
                  />
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  {notification.message}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {notification.time}
                </Typography>
              </Box>
            </MenuItem>
          ))}
          <Box sx={{ p: 2, textAlign: 'center', borderTop: '1px solid', borderColor: 'divider' }}>
            <Typography variant="body2" color="primary" sx={{ cursor: 'pointer' }}>
              View All Notifications
            </Typography>
          </Box>
        </Menu>

        {/* Profile Menu */}
        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={handleProfileMenuClose}
        >
          <MenuItem onClick={handleProfileMenuClose}>
            <AccountCircle sx={{ mr: 2 }} />
            Profile
          </MenuItem>
          <MenuItem onClick={handleProfileMenuClose}>
            <Settings sx={{ mr: 2 }} />
            Settings
          </MenuItem>
          <MenuItem onClick={handleProfileMenuClose}>
            <ExitToApp sx={{ mr: 2 }} />
            Logout
          </MenuItem>
        </Menu>
      </Toolbar>
    </AppBar>
  );
};

export default Header;