import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Typography,
  Box,
  Chip,
  Badge,
} from '@mui/material';
import {
  Dashboard,
  NetworkCheck,
  Public,
  Psychology,
  Link,
  Map,
  Analytics,
  Assignment,
  Description,
  Settings,
  Security,
  Warning,
} from '@mui/icons-material';

const DRAWER_WIDTH = 280;

const menuItems = [
  {
    text: 'Threat Overview',
    icon: <Dashboard />,
    path: '/',
    badge: null,
  },
  {
    text: 'Network Topology',
    icon: <NetworkCheck />,
    path: '/network-topology',
    badge: null,
  },
  {
    text: 'Threat Intelligence',
    icon: <Public />,
    path: '/threat-intelligence',
    badge: 'NEW',
  },
  {
    text: 'Behavioral Analysis',
    icon: <Psychology />,
    path: '/behavioral-analysis',
    badge: null,
  },
  {
    text: 'Attack Correlation',
    icon: <Link />,
    path: '/attack-correlation',
    badge: null,
  },
  {
    text: 'Geospatial Analysis',
    icon: <Map />,
    path: '/geospatial-analysis',
    badge: null,
  },
  {
    text: 'Advanced Analytics',
    icon: <Analytics />,
    path: '/advanced-analytics',
    badge: null,
  },
  {
    text: 'Incident Management',
    icon: <Assignment />,
    path: '/incident-management',
    badge: null,
  },
  {
    text: 'Reports & Export',
    icon: <Description />,
    path: '/reports-export',
    badge: null,
  },
  {
    text: 'Settings',
    icon: <Settings />,
    path: '/settings',
    badge: null,
  },
];

const Sidebar = ({ open, onToggle }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleNavigation = (path) => {
    navigate(path);
  };

  const drawerContent = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Logo and Title */}
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Security sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
        <Typography variant="h6" component="div" sx={{ fontWeight: 'bold' }}>
          NIDS Dashboard
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Network Intrusion Detection System
        </Typography>
      </Box>

      <Divider />

      {/* System Status */}
      <Box sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Box
            sx={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              backgroundColor: 'success.main',
              mr: 1,
            }}
          />
          <Typography variant="body2" color="text.secondary">
            System Online
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Warning sx={{ fontSize: 16, color: 'warning.main', mr: 1 }} />
          <Typography variant="body2" color="text.secondary">
            3 Active Threats
          </Typography>
        </Box>
      </Box>

      <Divider />

      {/* Navigation Menu */}
      <List sx={{ flexGrow: 1, px: 1 }}>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding sx={{ mb: 0.5 }}>
            <ListItemButton
              onClick={() => handleNavigation(item.path)}
              selected={location.pathname === item.path}
              sx={{
                borderRadius: 2,
                '&.Mui-selected': {
                  backgroundColor: 'primary.main',
                  color: 'primary.contrastText',
                  '&:hover': {
                    backgroundColor: 'primary.dark',
                  },
                  '& .MuiListItemIcon-root': {
                    color: 'primary.contrastText',
                  },
                },
                '&:hover': {
                  backgroundColor: 'action.hover',
                },
              }}
            >
              <ListItemIcon
                sx={{
                  minWidth: 40,
                  color: location.pathname === item.path ? 'inherit' : 'text.secondary',
                }}
              >
                {item.badge ? (
                  <Badge
                    badgeContent={item.badge}
                    color="secondary"
                    sx={{
                      '& .MuiBadge-badge': {
                        fontSize: '0.6rem',
                        height: 16,
                        minWidth: 16,
                      },
                    }}
                  >
                    {item.icon}
                  </Badge>
                ) : (
                  item.icon
                )}
              </ListItemIcon>
              <ListItemText
                primary={item.text}
                primaryTypographyProps={{
                  fontSize: '0.875rem',
                  fontWeight: location.pathname === item.path ? 600 : 400,
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider />

      {/* Quick Stats */}
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
          Quick Stats
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          <Chip
            label="24h: 156 alerts"
            size="small"
            color="primary"
            variant="outlined"
          />
          <Chip
            label="Critical: 3"
            size="small"
            color="error"
            variant="outlined"
          />
          <Chip
            label="Resolved: 89%"
            size="small"
            color="success"
            variant="outlined"
          />
        </Box>
      </Box>
    </Box>
  );

  return (
    <Drawer
      variant="persistent"
      anchor="left"
      open={open}
      sx={{
        width: open ? DRAWER_WIDTH : 0,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: DRAWER_WIDTH,
          boxSizing: 'border-box',
          borderRight: '1px solid',
          borderColor: 'divider',
          backgroundColor: 'background.paper',
        },
      }}
    >
      {drawerContent}
    </Drawer>
  );
};

export default Sidebar;